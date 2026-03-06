#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <omp.h>

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

// ==================== Bobot & Bias (definisi nyata) ====================
double weights_layer1[1400], biases_layer1[56];
double weights_layer2[672],  biases_layer2[12];
double weights_layer3[1296], biases_layer3[12];
double weights_layer4[1296], biases_layer4[12];
double weights_layer5[1296], biases_layer5[12];
double weights_layer6[1296], biases_layer6[12];
double weights_layer7[672],  biases_layer7[56];
double weights_layer8[4536], biases_layer8;

// ==================== Struktur Data dan Queue ====================
typedef struct {
    double *data;
    int frame_id;
    int rows, cols, channels;
} FrameBuffer;

typedef struct {
    FrameBuffer **buffer;
    int capacity;
    int head, tail, count;
    pthread_mutex_t mutex;
    pthread_cond_t cond_not_full;
    pthread_cond_t cond_not_empty;
} Queue;

Queue* create_queue(int capacity) {
    Queue *q = (Queue*)malloc(sizeof(Queue));
    q->buffer = (FrameBuffer**)malloc(capacity * sizeof(FrameBuffer*));
    q->capacity = capacity;
    q->head = q->tail = q->count = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->cond_not_full, NULL);
    pthread_cond_init(&q->cond_not_empty, NULL);
    return q;
}

void enqueue(Queue *q, FrameBuffer *item) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == q->capacity)
        pthread_cond_wait(&q->cond_not_full, &q->mutex);
    q->buffer[q->tail] = item;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    pthread_cond_signal(&q->cond_not_empty);
    pthread_mutex_unlock(&q->mutex);
}

FrameBuffer* dequeue(Queue *q) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0)
        pthread_cond_wait(&q->cond_not_empty, &q->mutex);
    FrameBuffer *item = q->buffer[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    pthread_cond_signal(&q->cond_not_full);
    pthread_mutex_unlock(&q->mutex);
    return item;
}

void destroy_queue(Queue *q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond_not_full);
    pthread_cond_destroy(&q->cond_not_empty);
    free(q->buffer);
    free(q);
}

// ==================== Buffer Alloc (langsung malloc/free) ====================
// Pool besar 363MB menyebabkan SIGTRAP di macOS; ganti dengan malloc/free biasa.
// Bottleneck tetap di operasi konvolusi, bukan alokasi memori.

double* get_buffer(int size) {
    double *p = (double*)malloc(size * sizeof(double));
    if (!p) { fprintf(stderr, "malloc gagal untuk %d double\n", size); exit(1); }
    return p;
}

void release_buffer(double *data) {
    free(data);
}


// ==================== Thread Stage ====================
typedef struct {
    int    layer_id;
    Queue *input_queue;
    Queue *output_queue;
    int    rows, cols, scale;
} StageArg;

void* stage_thread(void *arg) {
    StageArg *sa    = (StageArg*)arg;
    int rows_in     = sa->rows;
    int cols_in     = sa->cols;
    int scale       = sa->scale;
    int layer       = sa->layer_id;
    Queue *in_q     = sa->input_queue;
    Queue *out_q    = sa->output_queue;

    while (1) {
        FrameBuffer *in = dequeue(in_q);
        if (in->frame_id == -1) {       // sentinel: teruskan dan berhenti
            enqueue(out_q, in);
            break;
        }

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
        if (!out_data) {
            fprintf(stderr, "Gagal mendapatkan buffer untuk layer %d\n", layer);
            exit(1);
        }

        // Panggil fungsi layer
        switch (layer) {
            case 1: layer1(in->data, out_data, rows_in, cols_in); break;
            case 2: layer2(in->data, out_data, rows_in, cols_in); break;
            case 3: layer3(in->data, out_data, rows_in, cols_in); break;
            case 4: layer4(in->data, out_data, rows_in, cols_in); break;
            case 5: layer5(in->data, out_data, rows_in, cols_in); break;
            case 6: layer6(in->data, out_data, rows_in, cols_in); break;
            case 7: layer7(in->data, out_data, rows_in, cols_in); break;
            case 8: layer8(in->data, out_data, rows_in, cols_in, scale); break;
        }

        // Buat FrameBuffer output
        FrameBuffer *out_fb    = (FrameBuffer*)malloc(sizeof(FrameBuffer));
        out_fb->data           = out_data;
        out_fb->frame_id       = in->frame_id;
        out_fb->rows           = out_rows;
        out_fb->cols           = out_cols;
        out_fb->channels       = out_ch;

        release_buffer(in->data);
        free(in);

        enqueue(out_q, out_fb);
    }
    return NULL;
}

// ==================== Consumer Thread ====================
typedef struct {
    Queue          *q_out;
    FILE           *outFp;
    unsigned char **uv_store;
    int             uv_size;    // bytes per U atau V
    int             inRows, inCols;
    int             outRows, outCols;
    int            *frames_out; // counter (ditulis oleh consumer)
} ConsumerArg;

void* consumer_thread(void *arg) {
    ConsumerArg *ca   = (ConsumerArg*)arg;
    Queue *q_out      = ca->q_out;
    FILE  *outFp      = ca->outFp;
    int inRows = ca->inRows, inCols = ca->inCols;
    int outRows = ca->outRows, outCols = ca->outCols;
    int uv_size = ca->uv_size;

    unsigned char *hr_uint8 = (unsigned char*)malloc(outRows * outCols);
    unsigned char *outUBuf  = (unsigned char*)malloc((outCols/2) * (outRows/2));
    unsigned char *outVBuf  = (unsigned char*)malloc((outCols/2) * (outRows/2));

    while (1) {
        FrameBuffer *out = dequeue(q_out);
        if (out->frame_id == -1) { free(out); break; }

        int fid = out->frame_id;

        // Skala output FSRCNN [0,1] → [0,255] lalu konversi ke uint8
        int hr_pixels = outRows * outCols;
        for (int p = 0; p < hr_pixels; p++)
            out->data[p] *= 255.0;
        double_2_uint8(out->data, hr_uint8, outCols, outRows);
        fwrite(hr_uint8, 1, outRows * outCols, outFp);

        // Tulis U (replikasi 2x)
        unsigned char *uBuf = ca->uv_store[fid];
        unsigned char *vBuf = ca->uv_store[fid] + uv_size;
        for (int i = 0; i < inRows/2; i++)
        for (int j = 0; j < inCols/2; j++) {
            int cnt = 2 * (i * (outCols/2) + j);
            unsigned char u = uBuf[i * (inCols/2) + j];
            outUBuf[cnt]                   = u;
            outUBuf[cnt + 1]               = u;
            outUBuf[cnt + outCols/2]       = u;
            outUBuf[cnt + outCols/2 + 1]   = u;
        }
        fwrite(outUBuf, 1, (outCols/2)*(outRows/2), outFp);

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
        fwrite(outVBuf, 1, (outCols/2)*(outRows/2), outFp);

        release_buffer(out->data);
        free(out);
        (*ca->frames_out)++;
        printf("Frame %d selesai diproses.\n", fid + 1);
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
    // Kita simpan UV setiap frame supaya bisa ditulis secara per-frame
    // (setelah Y HR keluar dari pipeline) tanpa baca ulang file.
    int uv_size = (inCols / 2) * (inRows / 2);
    unsigned char **uv_store = (unsigned char**)malloc(numFrames * sizeof(unsigned char*));
    for (int f = 0; f < numFrames; f++)
        uv_store[f] = (unsigned char*)malloc(2 * uv_size); // U lalu V

    {
        unsigned char *yBuf = (unsigned char*)malloc(inCols * inRows);
        for (int f = 0; f < numFrames; f++) {
            if (fread(yBuf, 1, inCols * inRows, inFp) != (size_t)(inCols * inRows)) break;
            if (fread(uv_store[f], 1, 2 * uv_size, inFp) != (size_t)(2 * uv_size)) break;
        }
        free(yBuf);
    }
    // Kembalikan ke awal untuk membaca Y yang akan masuk pipeline
    rewind(inFp);

    // (pool diganti malloc/free langsung — tidak perlu inisialisasi)

    // ========== Buat queue antar layer ==========
    // 9 queue: input→L1→L2→L3→L4→L5→L6→L7→L8→output
    Queue *q_in  = create_queue(8); // main → layer1
    Queue *q12   = create_queue(8);
    Queue *q23   = create_queue(8);
    Queue *q34   = create_queue(8);
    Queue *q45   = create_queue(8);
    Queue *q56   = create_queue(8);
    Queue *q67   = create_queue(8);
    Queue *q78   = create_queue(8);
    Queue *q_out = create_queue(8); // layer8 → main

    // ========== Buat 8 thread stage ==========
    pthread_t t[8];
    StageArg args[8] = {
        {1, q_in, q12,   inRows, inCols, scale},
        {2, q12,  q23,   inRows, inCols, scale},
        {3, q23,  q34,   inRows, inCols, scale},
        {4, q34,  q45,   inRows, inCols, scale},
        {5, q45,  q56,   inRows, inCols, scale},
        {6, q56,  q67,   inRows, inCols, scale},
        {7, q67,  q78,   inRows, inCols, scale},
        {8, q78,  q_out, inRows, inCols, scale},
    };
    for (int i = 0; i < 8; i++)
        pthread_create(&t[i], NULL, stage_thread, &args[i]);

    // ========== Jalankan consumer thread (drains q_out ke file) ==========
    // PENTING: harus dijalankan SEBELUM loop feed supaya tidak terjadi deadlock.
    int frames_out = 0;
    ConsumerArg carg = {
        q_out, outFp, uv_store, uv_size,
        inRows, inCols, outRows, outCols, &frames_out
    };
    pthread_t t_consumer;
    pthread_create(&t_consumer, NULL, consumer_thread, &carg);

    // ========== Masukkan frame Y ke pipeline ==========
    unsigned char *inBuf = (unsigned char*)malloc(inCols * inRows);
    int frames_in = 0;
    for (frames_in = 0; frames_in < numFrames; frames_in++) {
        if (fread(inBuf, 1, inCols * inRows, inFp) != (size_t)(inCols * inRows)) break;
        fseek(inFp, 2 * uv_size, SEEK_CUR); // lewati UV (sudah di-store)

        double *lr_data = get_buffer(inRows * inCols);
        for (int i = 0; i < inRows * inCols; i++)
            lr_data[i] = inBuf[i] / 255.0;

        FrameBuffer *fb = (FrameBuffer*)malloc(sizeof(FrameBuffer));
        fb->data        = lr_data;
        fb->frame_id    = frames_in;
        fb->rows        = inRows;
        fb->cols        = inCols;
        fb->channels    = 1;
        enqueue(q_in, fb);
    }
    printf("Total %d frame dimasukkan ke pipeline.\n", frames_in);

    // Kirim sentinel
    FrameBuffer *sentinel = (FrameBuffer*)calloc(1, sizeof(FrameBuffer));
    sentinel->frame_id = -1;
    enqueue(q_in, sentinel);

    // Tunggu consumer selesai, lalu pipeline threads
    pthread_join(t_consumer, NULL);
    printf("Pipeline selesai, %d frame diproses.\n", frames_out);


    // ========== Tunggu semua thread ==========
    for (int i = 0; i < 8; i++)
        pthread_join(t[i], NULL);

    // ========== Bersihkan ==========
    free(inBuf);
    // hr_uint8, outUBuf, outVBuf dibebaskan di dalam consumer_thread

    for (int f = 0; f < numFrames; f++) free(uv_store[f]);
    free(uv_store);
    fclose(inFp);
    fclose(outFp);

    destroy_queue(q_in);
    destroy_queue(q12); destroy_queue(q23); destroy_queue(q34);
    destroy_queue(q45); destroy_queue(q56); destroy_queue(q67);
    destroy_queue(q78); destroy_queue(q_out);

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

    // VLA diganti malloc: VLA ~789KB meluap dari stack OMP thread (<256KB default)
    for (int j = 0; j < num_ch; j++) {
        double *img_fltr_8_tmp = (double*)malloc(hr_pixels * sizeof(double));
        if (!img_fltr_8_tmp) { free(accum); return; }
        deconv(input + j * rows * cols, img_fltr_8_tmp,
               weights_layer8 + j * filtersize, cols, rows, scale);
        // tanpa critical section — sama seperti kode fsrcnn_parallel.c asli
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
            *(uint8_img + cnt) = (unsigned char)(val + 0.5); // rounding
    }
}