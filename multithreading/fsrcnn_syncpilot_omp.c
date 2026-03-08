#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#include "framework/syncpilot.h"

// macOS tidak punya sched_getcpu()
#ifndef __linux__
#ifndef sched_getcpu
static inline int sched_getcpu(void) { return 0; }
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

// ==================== Buffer Alloc ====================
double* get_buffer(int size) {
    double *p = (double*)malloc(size * sizeof(double));
    if (!p) { fprintf(stderr, "malloc gagal untuk %d double\n", size); exit(1); }
    return p;
}

void release_buffer(double *data) {
    free(data);
}

// =========================================================
//  S T R U K T U R   K U S T O M   D A T A
// =========================================================
typedef struct {
    double *data;
    int    rows;
    int    cols;
    int    channels;
    int    scale;    // Tambahan informasi untuk layer8
} MyVideoFrame;


// =========================================================
//  F S R C N N  --  S T A G E S  -- (SYNCPILOT CALLBACKS)
// =========================================================

static void fsrcnn_process_stage(PipelineTask *task, int layer_num) {
    MyVideoFrame *fb = (MyVideoFrame*)task->data;
    double t_start = get_time();

    int cpu_id  = sched_getcpu();
    int rows_in = fb->rows;
    int cols_in = fb->cols;
    int scale   = fb->scale;

    int out_rows, out_cols, out_ch;
    switch (layer_num) {
        case 1: out_rows = rows_in; out_cols = cols_in; out_ch = 56; break;
        case 2: out_rows = rows_in; out_cols = cols_in; out_ch = 12; break;
        case 3: case 4: case 5:
        case 6: out_rows = rows_in; out_cols = cols_in; out_ch = 12; break;
        case 7: out_rows = rows_in; out_cols = cols_in; out_ch = 56; break;
        case 8: out_rows = rows_in * scale; out_cols = cols_in * scale; out_ch = 1; break;
        default: return; // error
    }

    int out_size = out_rows * out_cols * out_ch;
    double *out_data = get_buffer(out_size);

    switch (layer_num) {
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
    // Untuk WorkerID, secara native framework tidak nyimpen di parameter (krna generik), tapi output asinkron sgt cepat
    printf("[SYNCPILOT | CPU %2d] Layer %d memproses Frame %3d | Waktu: %.5f detik\n",
            cpu_id, layer_num, task->task_id + 1, t_end - t_start);

    // Update in-place ke data buatan user
    release_buffer(fb->data);
    fb->data     = out_data;
    fb->rows     = out_rows;
    fb->cols     = out_cols;
    fb->channels = out_ch;
}

// Wrapper khusus per layer (karena Framework menggunakan void(Task*)
void cb_layer1(PipelineTask *t) { fsrcnn_process_stage(t, 1); }
void cb_layer2(PipelineTask *t) { fsrcnn_process_stage(t, 2); }
void cb_layer3(PipelineTask *t) { fsrcnn_process_stage(t, 3); }
void cb_layer4(PipelineTask *t) { fsrcnn_process_stage(t, 4); }
void cb_layer5(PipelineTask *t) { fsrcnn_process_stage(t, 5); }
void cb_layer6(PipelineTask *t) { fsrcnn_process_stage(t, 6); }
void cb_layer7(PipelineTask *t) { fsrcnn_process_stage(t, 7); }
void cb_layer8(PipelineTask *t) { fsrcnn_process_stage(t, 8); }


// =========================================================
//  C O N S U M E R  --  P E N U L I S   A K H I R
// =========================================================
// Struct perantara agar Consumer punya pointer memori Video dan File
FILE *g_outFp = NULL;
unsigned char **g_uv_store = NULL;
int g_uv_size = 0, g_inRows = 0, g_inCols = 0, g_outRows = 0, g_outCols = 0;
int g_frames_out = 0;

void fsrcnn_consumer_writer(PipelineTask *task) {
    MyVideoFrame *fb = (MyVideoFrame*)task->data;
    int next = task->task_id;

    unsigned char *hr_uint8 = (unsigned char*)malloc(g_outRows * g_outCols);
    unsigned char *outUBuf  = (unsigned char*)malloc((g_outCols/2) * (g_outRows/2));
    unsigned char *outVBuf  = (unsigned char*)malloc((g_outCols/2) * (g_outRows/2));

    // Konversi Y [0,1] → [0,255] dan tulis
    int hr_pixels = g_outRows * g_outCols;
    for (int p = 0; p < hr_pixels; p++)
        fb->data[p] *= 255.0;
    
    double_2_uint8(fb->data, hr_uint8, g_outCols, g_outRows);
    fwrite(hr_uint8, 1, g_outRows * g_outCols, g_outFp);

    // Tulis U (replikasi 2x)
    unsigned char *uBuf = g_uv_store[next];
    unsigned char *vBuf = g_uv_store[next] + g_uv_size;
    for (int i = 0; i < g_inRows/2; i++)
    for (int j = 0; j < g_inCols/2; j++) {
        int cnt = 2 * (i * (g_outCols/2) + j);
        unsigned char u = uBuf[i * (g_inCols/2) + j];
        outUBuf[cnt]                   = u;
        outUBuf[cnt + 1]               = u;
        outUBuf[cnt + g_outCols/2]       = u;
        outUBuf[cnt + g_outCols/2 + 1]   = u;
    }
    fwrite(outUBuf, 1, (g_outCols/2)*(g_outRows/2), g_outFp);

    // Tulis V (replikasi 2x)
    for (int i = 0; i < g_inRows/2; i++)
    for (int j = 0; j < g_inCols/2; j++) {
        int cnt = 2 * (i * (g_outCols/2) + j);
        unsigned char v = vBuf[i * (g_inCols/2) + j];
        outVBuf[cnt]                   = v;
        outVBuf[cnt + 1]               = v;
        outVBuf[cnt + g_outCols/2]       = v;
        outVBuf[cnt + g_outCols/2 + 1]   = v;
    }
    fwrite(outVBuf, 1, (g_outCols/2)*(g_outRows/2), g_outFp);

    release_buffer(fb->data);

    g_frames_out++;
    printf("Frame %d selesai diproses.\n", next + 1);

    free(hr_uint8); free(outUBuf); free(outVBuf);
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

    // ========== Seting Consumer Global ==========
    g_inCols = inCols; g_inRows = inRows;
    g_outCols = outCols; g_outRows = outRows;

    // ========== Buka file ==========
    FILE *inFp = fopen(inFile, "rb");
    if (!inFp) { perror("fopen input"); return 1; }
    g_outFp = fopen(outFile, "wb");
    if (!g_outFp) { perror("fopen output"); fclose(inFp); return 1; }

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
    g_uv_size = uv_size;
    g_uv_store = (unsigned char**)malloc(numFrames * sizeof(unsigned char*));
    for (int f = 0; f < numFrames; f++)
        g_uv_store[f] = (unsigned char*)malloc(2 * uv_size);

    {
        unsigned char *yBuf = (unsigned char*)malloc(inCols * inRows);
        for (int f = 0; f < numFrames; f++) {
            if (fread(yBuf, 1, inCols * inRows, inFp) != (size_t)(inCols * inRows)) break;
            if (fread(g_uv_store[f], 1, 2 * uv_size, inFp) != (size_t)(2 * uv_size)) break;
        }
        free(yBuf);
    }
    rewind(inFp);


    // ==============================================================
    // 🔥 INISIALISASI SYNCPILOT FRAMEWORK 🔥
    // ==============================================================
    PipelineConfig cfg;
    cfg.num_workers              = 8;   // 8 pekerja paralel 
    cfg.num_stages               = 8;   // 8 layer FSRCNN
    cfg.total_tasks              = numFrames;
    cfg.queue_capacity_per_stage = 16;  // Sama seperti LAYER_Q_CAP = 16

    cfg.stages[0] = cb_layer1;
    cfg.stages[1] = cb_layer2;
    cfg.stages[2] = cb_layer3;
    cfg.stages[3] = cb_layer4;
    cfg.stages[4] = cb_layer5;
    cfg.stages[5] = cb_layer6;
    cfg.stages[6] = cb_layer7;
    cfg.stages[7] = cb_layer8;

    cfg.consumer = fsrcnn_consumer_writer; // Consumer Pengurut Akhir!

    printf("Memulai Engine SyncPilot...\n");
    PipelineEngine *engine = pipeline_start(&cfg);


    // ========== Feed frame Y ke Engine ==========
    unsigned char *inBuf = (unsigned char*)malloc(inCols * inRows);
    int frames_in = 0;
    for (frames_in = 0; frames_in < numFrames; frames_in++) {
        if (fread(inBuf, 1, inCols * inRows, inFp) != (size_t)(inCols * inRows)) break;
        fseek(inFp, 2 * uv_size, SEEK_CUR);

        double *lr_data = get_buffer(inRows * inCols);
        for (int i = 0; i < inRows * inCols; i++) {
            lr_data[i] = inBuf[i] / 255.0;
        }

        MyVideoFrame *baru = (MyVideoFrame*)malloc(sizeof(MyVideoFrame));
        baru->data     = lr_data;
        baru->rows     = inRows;
        baru->cols     = inCols;
        baru->channels = 1;
        baru->scale    = scale;

        // Push ke framework! (Automagic Load Balancing dimulai seketika)
        pipeline_feed(engine, frames_in, baru);
    }
    free(inBuf);
    printf("Total %d frame dimasukkan ke pipeline.\n", frames_in);

    // Kunci pintu feed
    pipeline_close_input(engine);

    // ========== Tunggu semua worker selesai ==========
    pipeline_wait_and_destroy(engine);

    printf("Pipeline SyncPilot selesai, %d frame ditulis ke disk.\n", g_frames_out);

    // ========== Bersihkan ==========
    for (int f = 0; f < numFrames; f++) free(g_uv_store[f]);
    free(g_uv_store);
    fclose(inFp);
    fclose(g_outFp);

    printf("Selesai.\n");
    return 0;
}


// ==================== IMPLEMENTASI LAYER ====================

void layer1(double *input, double *output, int rows, int cols) {
    const int filtersize  = 25;
    const int padsize     = 2;
    const int num_filters = 56;
    const double prelu    = -0.8986;
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
    for (int j = 0; j < num_ch; j++) {
        double *img_fltr_8_tmp = (double*)malloc(hr_pixels * sizeof(double));
        if (!img_fltr_8_tmp) continue;

        deconv(input + j * rows * cols, img_fltr_8_tmp,
               weights_layer8 + j * filtersize, cols, rows, scale);

        #pragma omp critical
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
