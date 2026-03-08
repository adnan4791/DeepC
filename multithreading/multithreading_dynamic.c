#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/stat.h>

FILE *log_file = NULL;

// macOS (terutama Apple Silicon) tidak mengekspos ID core fisik (CPU core ID) ke userspace (EL0).
// Sebagai alternatif untuk 'mengukur' atau membedakan load antar thread di log,
// kita mensimulasikan CPU ID secara logis menggunakan Thread-Local Storage (TLS).
#ifndef __linux__
#ifndef sched_getcpu
#include <stdatomic.h>
static inline int sched_getcpu(void) {
    static _Atomic int proxy_cpu_counter = 0;
    static __thread int my_proxy_cpu_id = -1;
    if (my_proxy_cpu_id == -1) {
        my_proxy_cpu_id = atomic_fetch_add(&proxy_cpu_counter, 1);
    }
    return my_proxy_cpu_id;
}
#endif
#endif

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// ==================== Forward Declarations ====================
void pad_image(double *img, double *img_pad, int rows, int cols, int padsize);
void imfilter(double *img, double *kernel, double *img_fltr, int rows, int cols, int padsize);
void PReLU(double *img_fltr, int rows, int cols, double bias, double prelu_coeff);
void deconv(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride);
void double_2_uint8(double *double_img, unsigned char *uint8_img, int cols, int rows);

void layer1(double *input, double *output, int rows, int cols);
void layer2(double *input, double *output, int rows, int cols);
void layer3(double *input, double *output, int rows, int cols);
void layer4(double *input, double *output, int rows, int cols);
void layer5(double *input, double *output, int rows, int cols);
void layer6(double *input, double *output, int rows, int cols);
void layer7(double *input, double *output, int rows, int cols);
void layer8(double *input, double *output, int rows, int cols, int scale);

// ==================== Bobot & Bias ====================
double weights_layer1[1400], biases_layer1[56];
double weights_layer2[672],  biases_layer2[12];
double weights_layer3[1296], biases_layer3[12];
double weights_layer4[1296], biases_layer4[12];
double weights_layer5[1296], biases_layer5[12];
double weights_layer6[1296], biases_layer6[12];
double weights_layer7[672],  biases_layer7[56];
double weights_layer8[4536], biases_layer8;

// ==================== FrameBuffer ====================
typedef struct {
    double *data;
    int frame_id;
    int rows, cols, channels;
} FrameBuffer;

// ==================== Buffer Alloc ====================
double* get_buffer(int size) {
    double *p = (double*)malloc(size * sizeof(double));
    if (!p) { fprintf(stderr, "malloc gagal untuk %d double\n", size); exit(1); }
    return p;
}

void release_buffer(double *data) {
    free(data);
}

// ==============================================================
//  DYNAMIC LOAD BALANCING — Worker Pool + Priority Task Queue
// ==============================================================
//
//  Arsitektur lama  : 1 thread tetap per layer (8 thread).
//                     Layer cepat (1,3-6) menganggur menunggu Layer lambat (8).
//
//  Arsitektur baru  : N worker thread yang SEMUA bisa memproses LAYER MANA SAJA.
//                     Worker mengambil pekerjaan dari antrean layer TERTINGGI dulu
//                     (Layer 8 → 7 → ... → 1).
//
//      Efek:
//      ┌──────────────────────────────────────────────────────────┐
//      │  Jika Layer 8 menumpuk, SEMUA idle worker akan membantu │
//      │  Layer 8 secara otomatis — tidak ada yang nganggur.     │
//      └──────────────────────────────────────────────────────────┘
//
//  Reorder Buffer   : Karena beberapa worker bisa menyelesaikan Layer 8
//                     untuk frame berbeda secara paralel, hasilnya mungkin
//                     tidak urut. ReorderBuffer memastikan penulisan file
//                     tetap berurutan (Frame 1, 2, 3, ...).
//
// ==============================================================

#define NUM_WORKERS   8     // jumlah worker (sesuaikan jumlah core CPU)
#define LAYER_Q_CAP  16    // kapasitas antrean per layer

// ---------- Small Queue (tanpa lock internal, dipakai dalam WorkPool) ----------
typedef struct {
    FrameBuffer *items[LAYER_Q_CAP];
    int head, tail, count;
} SmallQueue;

static int sq_push(SmallQueue *sq, FrameBuffer *fb) {
    if (sq->count >= LAYER_Q_CAP) return 0; // penuh
    sq->items[sq->tail] = fb;
    sq->tail = (sq->tail + 1) % LAYER_Q_CAP;
    sq->count++;
    return 1;
}

static FrameBuffer* sq_pop(SmallQueue *sq) {
    if (sq->count == 0) return NULL;
    FrameBuffer *fb = sq->items[sq->head];
    sq->head = (sq->head + 1) % LAYER_Q_CAP;
    sq->count--;
    return fb;
}

// ---------- Work Pool (antrean global untuk semua layer) ----------
typedef struct {
    SmallQueue      layer_q[8];     // index 0 = Layer 1, ... 7 = Layer 8
    pthread_mutex_t lock;
    pthread_cond_t  cond_work;      // sinyal: ada kerjaan baru
    pthread_cond_t  cond_space;     // sinyal: ada ruang di antrean
    int  tasks_in_flight;           // tugas yang sedang diproses worker
    int  input_done;                // semua frame sudah di-feed
    int  shutdown;                  // waktu berhenti
    int  rows, cols, scale;
} WorkPool;

// ---------- Reorder Buffer (mengurutkan output agar berurutan) ----------
typedef struct {
    FrameBuffer **slots;    // slots[frame_id] = hasil atau NULL
    int size;
    pthread_mutex_t lock;
    pthread_cond_t  cond;   // sinyal: ada output baru
} ReorderBuffer;

// ---------- Worker Thread ----------
typedef struct {
    WorkPool      *pool;
    ReorderBuffer *reorder;
    int            worker_id;
} WorkerArg;

void* worker_thread(void *arg) {
    WorkerArg *wa    = (WorkerArg*)arg;
    WorkPool  *pool  = wa->pool;
    ReorderBuffer *rb = wa->reorder;

    while (1) {
        pthread_mutex_lock(&pool->lock);

        // Cari kerjaan: prioritas Layer 8 (index 7) → Layer 1 (index 0)
        FrameBuffer *fb = NULL;
        int layer_idx   = -1;

        while (!pool->shutdown) {
            for (int l = 7; l >= 0; l--) {
                fb = sq_pop(&pool->layer_q[l]);
                if (fb) { layer_idx = l; break; }
            }
            if (fb) break;

            // Tidak ada kerjaan. Apakah semua sudah selesai?
            if (pool->input_done && pool->tasks_in_flight == 0) {
                int semua_kosong = 1;
                for (int i = 0; i < 8; i++) {
                    if (pool->layer_q[i].count > 0) { semua_kosong = 0; break; }
                }
                if (semua_kosong) {
                    pool->shutdown = 1;
                    pthread_cond_broadcast(&pool->cond_work);
                    break;
                }
            }
            // Tunggu sinyal kerjaan baru
            pthread_cond_wait(&pool->cond_work, &pool->lock);
        }

        if (pool->shutdown && !fb) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        pool->tasks_in_flight++;
        pthread_cond_signal(&pool->cond_space); // ruangan di antrean asal
        pthread_mutex_unlock(&pool->lock);

        // ====== Proses layer ======
        int layer    = layer_idx + 1; // layer_id 1-8
        int my_frame = fb->frame_id;
        int rows_in  = pool->rows;
        int cols_in  = pool->cols;
        int scale    = pool->scale;

        // Tentukan ukuran output
        int out_rows, out_cols, out_ch;
        switch (layer) {
            case 1:                  out_rows = rows_in;         out_cols = cols_in;         out_ch = 56; break;
            case 2:                  out_rows = rows_in;         out_cols = cols_in;         out_ch = 12; break;
            case 3: case 4: case 5:
            case 6:                  out_rows = rows_in;         out_cols = cols_in;         out_ch = 12; break;
            case 7:                  out_rows = rows_in;         out_cols = cols_in;         out_ch = 56; break;
            case 8:                  out_rows = rows_in * scale; out_cols = cols_in * scale; out_ch =  1; break;
            default:                 out_rows = out_cols = out_ch = 0;
        }
        int out_size = out_rows * out_cols * out_ch;
        double *out_data = get_buffer(out_size);

        double t_start = get_time();

        // Panggil fungsi layer
        switch (layer) {
            case 1: layer1(fb->data, out_data, rows_in, cols_in); break;
            case 2: layer2(fb->data, out_data, rows_in, cols_in); break;
            case 3: layer3(fb->data, out_data, rows_in, cols_in); break;
            case 4: layer4(fb->data, out_data, rows_in, cols_in); break;
            case 5: layer5(fb->data, out_data, rows_in, cols_in); break;
            case 6: layer6(fb->data, out_data, rows_in, cols_in); break;
            case 7: layer7(fb->data, out_data, rows_in, cols_in); break;
            case 8: layer8(fb->data, out_data, rows_in, cols_in, scale); break;
        }

        double t_end = get_time();
        if (log_file) {
            fprintf(log_file, "[Worker %d | CPU %2d] Layer %d memproses Frame %3d | Waktu: %.5f detik\n",
                   wa->worker_id, wa->worker_id, layer, my_frame + 1, t_end - t_start);
            fflush(log_file);
        }

        // Buat FrameBuffer output
        FrameBuffer *out_fb = (FrameBuffer*)malloc(sizeof(FrameBuffer));
        out_fb->data     = out_data;
        out_fb->frame_id = my_frame;
        out_fb->rows     = out_rows;
        out_fb->cols     = out_cols;
        out_fb->channels = out_ch;

        release_buffer(fb->data);
        free(fb);

        if (layer == 8) {
            // Tulis ke reorder buffer
            pthread_mutex_lock(&rb->lock);
            rb->slots[my_frame] = out_fb;
            pthread_cond_signal(&rb->cond);
            pthread_mutex_unlock(&rb->lock);

            pthread_mutex_lock(&pool->lock);
            pool->tasks_in_flight--;
            // Cek apakah semua selesai
            if (pool->input_done && pool->tasks_in_flight == 0) {
                int semua_kosong = 1;
                for (int i = 0; i < 8; i++) {
                    if (pool->layer_q[i].count > 0) { semua_kosong = 0; break; }
                }
                if (semua_kosong) {
                    pool->shutdown = 1;
                    pthread_cond_broadcast(&pool->cond_work);
                }
            }
            pthread_cond_signal(&pool->cond_space);
            pthread_mutex_unlock(&pool->lock);
        } else {
            // Masukkan ke antrean layer berikutnya
            int next_layer_idx = layer_idx + 1;
            pthread_mutex_lock(&pool->lock);
            while (!sq_push(&pool->layer_q[next_layer_idx], out_fb)) {
                // Antrean penuh, lepas lock agar worker lain bisa menguras
                pthread_cond_broadcast(&pool->cond_work);
                pthread_mutex_unlock(&pool->lock);
                usleep(100);  // serah CPU sebentar
                pthread_mutex_lock(&pool->lock);
            }
            pool->tasks_in_flight--;
            pthread_cond_broadcast(&pool->cond_work);
            pthread_cond_signal(&pool->cond_space);
            pthread_mutex_unlock(&pool->lock);
        }
    }
    return NULL;
}

// ==================== Consumer Thread (dengan Reorder Buffer) ====================
typedef struct {
    ReorderBuffer  *reorder;
    FILE           *outFp;
    unsigned char **uv_store;
    int             uv_size;
    int             inRows, inCols;
    int             outRows, outCols;
    int             total_frames;
    int            *frames_out;
} ConsumerArg;

void* consumer_thread(void *arg) {
    ConsumerArg *ca = (ConsumerArg*)arg;
    ReorderBuffer *rb = ca->reorder;
    int outRows = ca->outRows, outCols = ca->outCols;
    int inRows  = ca->inRows,  inCols  = ca->inCols;
    int uv_size = ca->uv_size;

    unsigned char *hr_uint8 = (unsigned char*)malloc(outRows * outCols);
    unsigned char *outUBuf  = (unsigned char*)malloc((outCols/2) * (outRows/2));
    unsigned char *outVBuf  = (unsigned char*)malloc((outCols/2) * (outRows/2));

    int next = 0;
    while (next < ca->total_frames) {
        // Tunggu frame berikutnya (next) tersedia di reorder buffer
        pthread_mutex_lock(&rb->lock);
        while (rb->slots[next] == NULL) {
            pthread_cond_wait(&rb->cond, &rb->lock);
        }
        FrameBuffer *out = rb->slots[next];
        rb->slots[next] = NULL;
        pthread_mutex_unlock(&rb->lock);

        // Konversi Y [0,1] → [0,255] dan tulis
        int hr_pixels = outRows * outCols;
        for (int p = 0; p < hr_pixels; p++)
            out->data[p] *= 255.0;
        double_2_uint8(out->data, hr_uint8, outCols, outRows);
        fwrite(hr_uint8, 1, outRows * outCols, ca->outFp);

        // Tulis U (replikasi 2x)
        unsigned char *uBuf = ca->uv_store[next];
        unsigned char *vBuf = ca->uv_store[next] + uv_size;
        for (int i = 0; i < inRows/2; i++)
        for (int j = 0; j < inCols/2; j++) {
            int cnt = 2 * (i * (outCols/2) + j);
            unsigned char u = uBuf[i * (inCols/2) + j];
            outUBuf[cnt]                   = u;
            outUBuf[cnt + 1]               = u;
            outUBuf[cnt + outCols/2]       = u;
            outUBuf[cnt + outCols/2 + 1]   = u;
        }
        fwrite(outUBuf, 1, (outCols/2)*(outRows/2), ca->outFp);

        // Tulis V (replikasi 2x)
        for (int i = 0; i < inRows/2; i++)
        for (int j = 0; j < inCols/2; j++) {
            int cnt = 2 * (i * (outCols/2) + j);
            unsigned char v = vBuf[i * (inCols/2) + j];
            outVBuf[cnt]                   = v;
            outVBuf[cnt + 1]               = v;
            outVBuf[cnt + outCols/2]       = v;
            outVBuf[cnt + outCols/2 + 1]   = v;
        }
        fwrite(outVBuf, 1, (outCols/2)*(outRows/2), ca->outFp);

        release_buffer(out->data);
        free(out);
        (*ca->frames_out)++;
        printf("Frame %d selesai diproses.\n", next + 1);
        next++;
    }

    free(hr_uint8); free(outUBuf); free(outVBuf);
    return NULL;
}


// ==================== MAIN ====================
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.yuv output.yuv\n", argv[0]);
        return 1;
    }
    char *inFile  = argv[1];
    char *outFile = argv[2];

    mkdir("logs", 0777);
    log_file = fopen("logs/multitreahding_dinamic.txt", "w");

    const int scale     = 2;
    const int inCols    = 176;
    const int inRows    = 144;
    const int outCols   = inCols * scale;
    const int outRows   = inRows * scale;
    const int numFrames = 150;

    // ========== Buka file ==========
    FILE *inFp = fopen(inFile, "rb");
    if (!inFp) { perror("fopen input"); return 1; }
    FILE *outFp = fopen(outFile, "wb");
    if (!outFp) { perror("fopen output"); fclose(inFp); return 1; }

    // ========== Muat bobot & bias ==========
    FILE *fp;
    #define LOAD_W(file, arr, n) \
        fp = fopen(file, "r"); \
        if (!fp) { printf("Error: %s\n", file); return 1; } \
        for (int _i = 0; _i < (n); _i++) fscanf(fp, "%lf", &(arr)[_i]); \
        fclose(fp);

    LOAD_W("weights_layer1.txt", weights_layer1, 1400)
    LOAD_W("biasess_layer1.txt", biases_layer1,   56)
    LOAD_W("weights_layer2.txt", weights_layer2,  672)
    LOAD_W("biasess_layer2.txt", biases_layer2,    12)
    LOAD_W("weights_layer3.txt", weights_layer3, 1296)
    LOAD_W("biasess_layer3.txt", biases_layer3,    12)
    LOAD_W("weights_layer4.txt", weights_layer4, 1296)
    LOAD_W("biasess_layer4.txt", biases_layer4,    12)
    LOAD_W("weights_layer5.txt", weights_layer5, 1296)
    LOAD_W("biasess_layer5.txt", biases_layer5,    12)
    LOAD_W("weights_layer6.txt", weights_layer6, 1296)
    LOAD_W("biasess_layer6.txt", biases_layer6,    12)
    LOAD_W("weights_layer7.txt", weights_layer7,  672)
    LOAD_W("biasess_layer7.txt", biases_layer7,    56)
    LOAD_W("weights_layer8.txt", weights_layer8, 4536)
    fp = fopen("biasess_layer8.txt", "r");
    if (!fp) { printf("Error: biasess_layer8.txt\n"); return 1; }
    fscanf(fp, "%lf", &biases_layer8);
    fclose(fp);

    printf("Bobot & bias berhasil dimuat.\n");

    // ========== Pre-baca UV semua frame ==========
    int uv_size = (inCols / 2) * (inRows / 2);
    unsigned char **uv_store = (unsigned char**)malloc(numFrames * sizeof(unsigned char*));
    for (int f = 0; f < numFrames; f++)
        uv_store[f] = (unsigned char*)malloc(2 * uv_size);

    {
        unsigned char *yBuf = (unsigned char*)malloc(inCols * inRows);
        for (int f = 0; f < numFrames; f++) {
            if (fread(yBuf, 1, inCols * inRows, inFp) != (size_t)(inCols * inRows)) break;
            if (fread(uv_store[f], 1, 2 * uv_size, inFp) != (size_t)(2 * uv_size)) break;
        }
        free(yBuf);
    }
    rewind(inFp);

    // ========== Buat Work Pool ==========
    WorkPool pool;
    memset(&pool, 0, sizeof(pool));
    pthread_mutex_init(&pool.lock, NULL);
    pthread_cond_init(&pool.cond_work, NULL);
    pthread_cond_init(&pool.cond_space, NULL);
    pool.rows  = inRows;
    pool.cols  = inCols;
    pool.scale = scale;

    // ========== Buat Reorder Buffer ==========
    ReorderBuffer reorder;
    reorder.slots = (FrameBuffer**)calloc(numFrames, sizeof(FrameBuffer*));
    reorder.size  = numFrames;
    pthread_mutex_init(&reorder.lock, NULL);
    pthread_cond_init(&reorder.cond, NULL);

    // ========== Jalankan Consumer Thread ==========
    int frames_out = 0;
    ConsumerArg carg = {
        &reorder, outFp, uv_store, uv_size,
        inRows, inCols, outRows, outCols, numFrames, &frames_out
    };
    pthread_t t_consumer;
    pthread_create(&t_consumer, NULL, consumer_thread, &carg);

    // ========== Jalankan Worker Threads ==========
    printf("Menjalankan %d worker thread (dynamic load balancing)...\n", NUM_WORKERS);
    pthread_t workers[NUM_WORKERS];
    WorkerArg wargs[NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; i++) {
        wargs[i].pool      = &pool;
        wargs[i].reorder   = &reorder;
        wargs[i].worker_id = i;
        pthread_create(&workers[i], NULL, worker_thread, &wargs[i]);
    }

    // ========== Feed frame Y ke pool (layer_q[0] = antrean Layer 1) ==========
    unsigned char *inBuf = (unsigned char*)malloc(inCols * inRows);
    int frames_in = 0;
    for (frames_in = 0; frames_in < numFrames; frames_in++) {
        if (fread(inBuf, 1, inCols * inRows, inFp) != (size_t)(inCols * inRows)) break;
        fseek(inFp, 2 * uv_size, SEEK_CUR);

        double *lr_data = get_buffer(inRows * inCols);
        for (int i = 0; i < inRows * inCols; i++)
            lr_data[i] = inBuf[i] / 255.0;

        FrameBuffer *fb = (FrameBuffer*)malloc(sizeof(FrameBuffer));
        fb->data     = lr_data;
        fb->frame_id = frames_in;
        fb->rows     = inRows;
        fb->cols     = inCols;
        fb->channels = 1;

        // Masukkan ke antrean Layer 1, tunggu jika penuh (backpressure)
        pthread_mutex_lock(&pool.lock);
        while (!sq_push(&pool.layer_q[0], fb)) {
            pthread_cond_wait(&pool.cond_space, &pool.lock);
        }
        pthread_cond_broadcast(&pool.cond_work);
        pthread_mutex_unlock(&pool.lock);
    }
    printf("Total %d frame dimasukkan ke pipeline.\n", frames_in);

    // Tandai semua input sudah masuk
    pthread_mutex_lock(&pool.lock);
    pool.input_done = 1;
    pthread_cond_broadcast(&pool.cond_work);
    pthread_mutex_unlock(&pool.lock);

    // ========== Tunggu semua worker selesai ==========
    for (int i = 0; i < NUM_WORKERS; i++)
        pthread_join(workers[i], NULL);

    // Tunggu consumer selesai menulis semua frame
    pthread_join(t_consumer, NULL);

    printf("Pipeline selesai, %d frame diproses.\n", frames_out);

    // ========== Bersihkan ==========
    free(inBuf);
    for (int f = 0; f < numFrames; f++) free(uv_store[f]);
    free(uv_store);
    free(reorder.slots);
    pthread_mutex_destroy(&reorder.lock);
    pthread_cond_destroy(&reorder.cond);
    pthread_mutex_destroy(&pool.lock);
    pthread_cond_destroy(&pool.cond_work);
    pthread_cond_destroy(&pool.cond_space);
    fclose(inFp);
    fclose(outFp);

    printf("Selesai.\n");
    return 0;
}


// ==================== IMPLEMENTASI LAYER ====================

void layer1(double *input, double *output, int rows, int cols) {
    const int filtersize  = 25;
    const int padsize     = 2;
    const int num_filters = 56;
    const double prelu    = -0.8986;
    for (int i = 0; i < num_filters; i++) {
        imfilter(input, weights_layer1 + i * filtersize,
                 output + i * rows * cols, rows, cols, padsize);
        PReLU(output + i * rows * cols, rows, cols, biases_layer1[i], prelu);
    }
}


void layer2(double *input, double *output, int rows, int cols) {
    const int filtersize  = 1;
    const int padsize     = 0;
    const int num_filters = 12;
    const int num_ch      = 56;
    const double prelu    = 0.3236;
    memset(output, 0, rows * cols * num_filters * sizeof(double));
    for (int i = 0; i < num_filters; i++) {

        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_ch; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer2 + (i * num_ch + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++)
                output[i * rows * cols + p] += tmp[p];
        }
        PReLU(output + i * rows * cols, rows, cols, biases_layer2[i], prelu);
        free(tmp);
    }
}

void layer3(double *input, double *output, int rows, int cols) {
    const int filtersize  = 9;
    const int padsize     = 1;
    const int num_filters = 12;
    const int num_ch      = 12;
    const double prelu    = 0.2288;
    memset(output, 0, rows * cols * num_filters * sizeof(double));
    for (int i = 0; i < num_filters; i++) {

        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_ch; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer3 + (i * num_ch + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++)
                output[i * rows * cols + p] += tmp[p];
        }
        PReLU(output + i * rows * cols, rows, cols, biases_layer3[i], prelu);
        free(tmp);
    }
}

void layer4(double *input, double *output, int rows, int cols) {
    const int filtersize  = 9;
    const int padsize     = 1;
    const int num_filters = 12;
    const int num_ch      = 12;
    const double prelu    = 0.2476;
    memset(output, 0, rows * cols * num_filters * sizeof(double));
    for (int i = 0; i < num_filters; i++) {

        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_ch; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer4 + (i * num_ch + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++)
                output[i * rows * cols + p] += tmp[p];
        }
        PReLU(output + i * rows * cols, rows, cols, biases_layer4[i], prelu);
        free(tmp);
    }
}

void layer5(double *input, double *output, int rows, int cols) {
    const int filtersize  = 9;
    const int padsize     = 1;
    const int num_filters = 12;
    const int num_ch      = 12;
    const double prelu    = 0.3495;
    memset(output, 0, rows * cols * num_filters * sizeof(double));
    for (int i = 0; i < num_filters; i++) {

        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_ch; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer5 + (i * num_ch + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++)
                output[i * rows * cols + p] += tmp[p];
        }
        PReLU(output + i * rows * cols, rows, cols, biases_layer5[i], prelu);
        free(tmp);
    }
}

void layer6(double *input, double *output, int rows, int cols) {
    const int filtersize  = 9;
    const int padsize     = 1;
    const int num_filters = 12;
    const int num_ch      = 12;
    const double prelu    = 0.7806;
    memset(output, 0, rows * cols * num_filters * sizeof(double));
    for (int i = 0; i < num_filters; i++) {

        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_ch; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer6 + (i * num_ch + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++)
                output[i * rows * cols + p] += tmp[p];
        }
        PReLU(output + i * rows * cols, rows, cols, biases_layer6[i], prelu);
        free(tmp);
    }
}

void layer7(double *input, double *output, int rows, int cols) {
    const int filtersize  = 1;
    const int padsize     = 0;
    const int num_filters = 56;
    const int num_ch      = 12;
    const double prelu    = 0.0087;
    memset(output, 0, rows * cols * num_filters * sizeof(double));
    for (int i = 0; i < num_filters; i++) {

        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_ch; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer7 + (i * num_ch + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++)
                output[i * rows * cols + p] += tmp[p];
        }
        PReLU(output + i * rows * cols, rows, cols, biases_layer7[i], prelu);
        free(tmp);
    }
}

void imadd(double *img_fltr_sum, double *img_fltr_crnt, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
        int cnt = i * cols + j;
        *(img_fltr_sum + cnt) = *(img_fltr_sum + cnt) + *(img_fltr_crnt + cnt);
    }
}

void layer8(double *input, double *output, int rows, int cols, int scale) {
    const int filtersize = 81; // 9x9
    const int num_ch     = 56;
    int hr_pixels        = (rows * scale) * (cols * scale);

    double *accum = (double*)calloc(hr_pixels, sizeof(double));
    if (!accum) return;

    for (int j = 0; j < num_ch; j++) {
        double *img_fltr_8_tmp = (double*)malloc(hr_pixels * sizeof(double));
        if (!img_fltr_8_tmp) { free(accum); return; }
        deconv(input + j * rows * cols, img_fltr_8_tmp,
               weights_layer8 + j * filtersize, cols, rows, scale);
        imadd(accum, img_fltr_8_tmp, cols * scale, rows * scale);
        free(img_fltr_8_tmp);
    }

    for (int i = 0; i < rows * scale; i++)
    for (int j = 0; j < cols * scale; j++) {
        int cnt = i * cols * scale + j;
        output[cnt] = accum[cnt] + biases_layer8;
    }

    free(accum);
}


// ==================== FUNGSI HELPER ====================

void pad_image(double *img, double *img_pad, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    int i, j, k, cnt, cnt_pad, k1, k2;
    for (i = padsize; i < rows_pad - padsize; i++)
    for (j = padsize; j < cols_pad - padsize; j++) {
        cnt_pad = i * cols_pad + j;
        cnt     = (i - padsize) * cols + j - padsize;
        *(img_pad + cnt_pad) = *(img + cnt);
    }
    for (j = padsize; j < cols_pad - padsize; j++)
    for (k = 0; k < padsize; k++) {
        cnt_pad = j + k * cols_pad;
        cnt     = j - padsize;
        *(img_pad + cnt_pad) = *(img + cnt);
        cnt_pad = j + (rows_pad - 1 - k) * cols_pad;
        cnt     = (j - padsize) + (rows - 1) * cols;
        *(img_pad + cnt_pad) = *(img + cnt);
    }
    for (i = padsize; i < rows_pad - padsize; i++)
    for (k = 0; k < padsize; k++) {
        cnt     = (i - padsize) * cols;
        cnt_pad = i * cols_pad + k;
        *(img_pad + cnt_pad) = *(img + cnt);
        cnt     = (i - padsize) * cols + cols - 1;
        cnt_pad = i * cols_pad + cols_pad - 1 - k;
        *(img_pad + cnt_pad) = *(img + cnt);
    }
    for (k1 = 0; k1 < padsize; k1++)
    for (k2 = 0; k2 < padsize; k2++) {
        *(img_pad + k1 * cols_pad + k2)                               = *(img);
        *(img_pad + k1 * cols_pad + cols_pad - 1 - k2)                = *(img + cols - 1);
        *(img_pad + (rows_pad-1-k1) * cols_pad + k2)                  = *(img + (rows-1)*cols);
        *(img_pad + (rows_pad-1-k1) * cols_pad + cols_pad - 1 - k2)   = *(img + (rows-1)*cols + cols-1);
    }
}

void imfilter(double *img, double *kernel, double *img_fltr, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    double *img_pad = (double*)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image(img, img_pad, rows, cols, padsize);
    for (int i = padsize; i < rows_pad - padsize; i++)
    for (int j = padsize; j < cols_pad - padsize; j++) {
        int cnt = (i - padsize) * cols + (j - padsize);
        double sum = 0.0;
        int cnt_krnl = 0;
        for (int k1 = -padsize; k1 <= padsize; k1++)
        for (int k2 = -padsize; k2 <= padsize; k2++) {
            int cnt_pad = (i + k1) * cols_pad + j + k2;
            sum += (*(img_pad + cnt_pad)) * (*(kernel + cnt_krnl));
            cnt_krnl++;
        }
        *(img_fltr + cnt) = sum;
    }
    free(img_pad);
}

static double _max2(double a, double b) { return a > b ? a : b; }
static double _min2(double a, double b) { return a > b ? b : a; }

void PReLU(double *img_fltr, int rows, int cols, double bias, double prelu_coeff)
{
    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
        int cnt = i * cols + j;
        double v = *(img_fltr + cnt) + bias;
        *(img_fltr + cnt) = _max2(v, 0.0) + prelu_coeff * _min2(v, 0.0);
    }
}

void deconv(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride)
{
    int border = 1, fsize = 9;
    int rows_pad = rows + 2 * border;
    int cols_pad = cols + 2 * border;
    double *img_input_padded = (double*)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image(img_input, img_input_padded, rows, cols, border);

    int rows_out_pad = rows_pad * stride;
    int cols_out_pad = cols_pad * stride;
    double *img_output_tmp = (double*)calloc((rows_out_pad + fsize - 1) * (cols_out_pad + fsize - 1), sizeof(double));
    double *kernel_modif   = (double*)malloc(fsize * fsize * sizeof(double));

    for (int i = 0; i < rows_pad; i++)
    for (int j = 0; j < cols_pad; j++) {
        int cnt_img        = i * cols_pad + j;
        int cnt_img_output = (i * stride) * (cols_out_pad + fsize - 1) + (j * stride);
        for (int k_r = 0; k_r < fsize; k_r++) {
            for (int k_c = 0; k_c < fsize; k_c++) {
                int ck = k_r * fsize + k_c;
                kernel_modif[ck] = kernel[ck] * img_input_padded[cnt_img];
                img_output_tmp[cnt_img_output + k_c] += kernel_modif[ck];
            }
            cnt_img_output += cols_out_pad + fsize - 1;
        }
    }

    int rows_out = rows * stride, cols_out = cols * stride;
    for (int i = 0; i < rows_out; i++)
    for (int j = 0; j < cols_out; j++) {
        int i_tmp = i + ((fsize + 1) / 2) + stride * border - 1;
        int j_tmp = j + ((fsize + 1) / 2) + stride * border - 1;
        img_output[i * cols_out + j] = img_output_tmp[i_tmp * (cols_out_pad + fsize - 1) + j_tmp];
    }

    free(img_input_padded);
    free(img_output_tmp);
    free(kernel_modif);
}

void double_2_uint8(double *double_img, unsigned char *uint8_img, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
        int cnt    = i * cols + j;
        double val = *(double_img + cnt);
        if (val <= 0.0)
            *(uint8_img + cnt) = 0;
        else if (val >= 255.0)
            *(uint8_img + cnt) = 255;
        else
            *(uint8_img + cnt) = (unsigned char)(val + 0.5);
    }
}