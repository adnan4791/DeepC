#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>

// ==================== Deklarasi Fungsi Helper ====================
void imfilter(double *img, double *kernel, double *img_fltr, int rows, int cols, int padsize);
void PReLU(double *img_fltr, int rows, int cols, double bias, double prelu_coeff);
void deconv(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride);

// ==================== Deklarasi Fungsi Layer ====================
// (Diambil dari kode asli, tetapi dipisah per layer)
void layer1(double *input, double *output, int rows, int cols, int num_threads);
void layer2(double *input, double *output, int rows, int cols, int num_threads);
void layer3(double *input, double *output, int rows, int cols, int num_threads);
void layer4(double *input, double *output, int rows, int cols, int num_threads);
void layer5(double *input, double *output, int rows, int cols, int num_threads);
void layer6(double *input, double *output, int rows, int cols, int num_threads);
void layer7(double *input, double *output, int rows, int cols, int num_threads);
void layer8(double *input, double *output, int rows, int cols, int scale, int num_threads);

// Bobot dan bias — definisi nyata (dimuat dari file .txt di main)
double weights_layer1[1400], biases_layer1[56];
double weights_layer2[672],  biases_layer2[12];
double weights_layer3[1296], biases_layer3[12];
double weights_layer4[1296], biases_layer4[12];
double weights_layer5[1296], biases_layer5[12];
double weights_layer6[1296], biases_layer6[12];
double weights_layer7[672],  biases_layer7[56];
double weights_layer8[4536], biases_layer8;

// ==================== Fungsi Bantu ====================
void set_affinity(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("sched_setaffinity");
    }
#else
    (void)core_id; // CPU affinity not supported on this platform
#endif
}

double profile_layer(void (*layer_func)(double*, double*, int, int, int),
                     double *input, double *output,
                     int rows, int cols, int num_threads, int core_base) {
    // Set afinitas untuk thread utama
    set_affinity(core_base);
    
    double start = omp_get_wtime();
    layer_func(input, output, rows, cols, num_threads);
    double end = omp_get_wtime();
    
    return end - start;
}

// Untuk layer 8 yang punya parameter scale
double profile_layer8(double *input, double *output,
                      int rows, int cols, int scale,
                      int num_threads, int core_base) {
    set_affinity(core_base);
    
    double start = omp_get_wtime();
    layer8(input, output, rows, cols, scale, num_threads);
    double end = omp_get_wtime();
    
    return end - start;
}

// ==================== Inisialisasi Data Dummy ====================
void init_data(double *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (double)rand() / RAND_MAX;  // nilai acak 0-1
    }
}

// ==================== MAIN ====================
int main() {
    int inRows = 144, inCols = 176;
    int scale = 2;
    int outRows = inRows * scale, outCols = inCols * scale;
    
    // Alokasi buffer
    // temp dan layer_output harus cukup untuk output terbesar: 56 * inRows * inCols (layer 1 & 7)
    // dan juga cukup untuk HR output layer 8: outRows * outCols
    int max_buf = inRows * inCols * 56;
    if (outRows * outCols > max_buf) max_buf = outRows * outCols;
    double *img_lr = (double*)malloc(inRows * inCols * sizeof(double));
    double *layer_output = (double*)malloc(max_buf * sizeof(double));
    double *temp = (double*)malloc(max_buf * sizeof(double));
    
    if (!img_lr || !layer_output || !temp) {
        printf("Alokasi memori gagal!\n");
        return 1;
    }
    
    // Inisialisasi data input acak
    init_data(img_lr, inRows * inCols);

    // ==================== Muat Bobot dan Bias dari File ====================
    FILE *fp;
    #define LOAD_WEIGHTS(file, arr, n) \
        fp = fopen(file, "r"); \
        if (!fp) { printf("Error membuka %s\n", file); return 1; } \
        for (int _i = 0; _i < (n); _i++) fscanf(fp, "%lf", &(arr)[_i]); \
        fclose(fp);

    LOAD_WEIGHTS("weights_layer1.txt", weights_layer1, 1400)
    LOAD_WEIGHTS("biasess_layer1.txt", biases_layer1, 56)
    LOAD_WEIGHTS("weights_layer2.txt", weights_layer2, 672)
    LOAD_WEIGHTS("biasess_layer2.txt", biases_layer2, 12)
    LOAD_WEIGHTS("weights_layer3.txt", weights_layer3, 1296)
    LOAD_WEIGHTS("biasess_layer3.txt", biases_layer3, 12)
    LOAD_WEIGHTS("weights_layer4.txt", weights_layer4, 1296)
    LOAD_WEIGHTS("biasess_layer4.txt", biases_layer4, 12)
    LOAD_WEIGHTS("weights_layer5.txt", weights_layer5, 1296)
    LOAD_WEIGHTS("biasess_layer5.txt", biases_layer5, 12)
    LOAD_WEIGHTS("weights_layer6.txt", weights_layer6, 1296)
    LOAD_WEIGHTS("biasess_layer6.txt", biases_layer6, 12)
    LOAD_WEIGHTS("weights_layer7.txt", weights_layer7, 672)
    LOAD_WEIGHTS("biasess_layer7.txt", biases_layer7, 56)
    LOAD_WEIGHTS("weights_layer8.txt", weights_layer8, 4536)
    fp = fopen("biasess_layer8.txt", "r");
    if (!fp) { printf("Error membuka biasess_layer8.txt\n"); return 1; }
    fscanf(fp, "%lf", &biases_layer8);
    fclose(fp);


    printf("========================================\n");
    printf("PROFILING LAYER FSRCNN\n");
    printf("========================================\n");
    
    // Konfigurasi pengujian
    int thread_counts[] = {1, 2, 3, 4};
    int num_thread_configs = sizeof(thread_counts) / sizeof(int);
    
    // Big cores: 4-7, LITTLE cores: 0-3
    int big_base = 4;
    int little_base = 0;
    
    // Layer 1
    printf("\n--- Layer 1 (Conv 5x5) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        // Big
        double time_big = profile_layer(layer1, img_lr, temp, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        // LITTLE
        double time_little = profile_layer(layer1, img_lr, temp, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 2
    printf("\n--- Layer 2 (Conv 1x1) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer(layer2, temp, layer_output, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer(layer2, temp, layer_output, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 3
    printf("\n--- Layer 3 (Conv 3x3) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer(layer3, layer_output, temp, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer(layer3, layer_output, temp, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 4
    printf("\n--- Layer 4 (Conv 3x3) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer(layer4, temp, layer_output, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer(layer4, temp, layer_output, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 5
    printf("\n--- Layer 5 (Conv 3x3) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer(layer5, layer_output, temp, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer(layer5, layer_output, temp, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 6
    printf("\n--- Layer 6 (Conv 3x3) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer(layer6, temp, layer_output, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer(layer6, temp, layer_output, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 7
    printf("\n--- Layer 7 (Conv 1x1) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer(layer7, layer_output, temp, inRows, inCols, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer(layer7, layer_output, temp, inRows, inCols, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    // Layer 8 (deconv)
    printf("\n--- Layer 8 (Deconv 9x9) ---\n");
    for (int t = 0; t < num_thread_configs; t++) {
        int nt = thread_counts[t];
        double time_big = profile_layer8(temp, layer_output, inRows, inCols, scale, nt, big_base);
        printf("  %d thread (big)   : %.6f s\n", nt, time_big);
        double time_little = profile_layer8(temp, layer_output, inRows, inCols, scale, nt, little_base);
        printf("  %d thread (LITTLE): %.6f s\n", nt, time_little);
    }
    
    printf("\n========================================\n");
    printf("PROFILING SELESAI\n");
    
    free(img_lr);
    free(layer_output);
    free(temp);
    return 0;
}

// ==================== LAYER 1 ====================
void layer1(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 25;
    int padsize = 2;
    int num_filters = 56;
    double prelu_coeff = -0.8986;

    omp_set_num_threads(num_threads);
    // output: rows * cols * num_filters
    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        imfilter(input, weights_layer1 + i * filtersize,
                 output + i * rows * cols, rows, cols, padsize);
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer1[i], prelu_coeff);
    }
}

// ==================== LAYER 2 ====================
void layer2(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 1;
    int padsize = 0;
    int num_filters = 12;
    int num_channels = 56;
    double prelu_coeff = 0.3236;

    omp_set_num_threads(num_threads);
    // output: rows * cols * num_filters, diinisialisasi nol
    memset(output, 0, rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_channels; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer2 + (i * num_channels + j) * filtersize,
                     tmp, rows, cols, padsize);
            // imadd ke output
            for (int p = 0; p < rows * cols; p++) {
                output[i * rows * cols + p] += tmp[p];
            }
        }
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer2[i], prelu_coeff);
        free(tmp);
    }
}

// ==================== LAYER 3 (mirip layer 2, dengan kernel 3x3) ====================
void layer3(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 9; // 3x3
    int padsize = 1;
    int num_filters = 12;
    int num_channels = 12;
    double prelu_coeff = 0.2288;

    omp_set_num_threads(num_threads);
    memset(output, 0, rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_channels; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer3 + (i * num_channels + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++) {
                output[i * rows * cols + p] += tmp[p];
            }
        }
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer3[i], prelu_coeff);
        free(tmp);
    }
}

// ==================== LAYER 4 ====================
void layer4(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 9;
    int padsize = 1;
    int num_filters = 12;
    int num_channels = 12;
    double prelu_coeff = 0.2476;

    omp_set_num_threads(num_threads);
    memset(output, 0, rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_channels; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer4 + (i * num_channels + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++) {
                output[i * rows * cols + p] += tmp[p];
            }
        }
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer4[i], prelu_coeff);
        free(tmp);
    }
}

// ==================== LAYER 5 ====================
void layer5(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 9;
    int padsize = 1;
    int num_filters = 12;
    int num_channels = 12;
    double prelu_coeff = 0.3495;

    omp_set_num_threads(num_threads);
    memset(output, 0, rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_channels; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer5 + (i * num_channels + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++) {
                output[i * rows * cols + p] += tmp[p];
            }
        }
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer5[i], prelu_coeff);
        free(tmp);
    }
}

// ==================== LAYER 6 ====================
void layer6(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 9;
    int padsize = 1;
    int num_filters = 12;
    int num_channels = 12;
    double prelu_coeff = 0.7806;

    omp_set_num_threads(num_threads);
    memset(output, 0, rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_channels; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer6 + (i * num_channels + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++) {
                output[i * rows * cols + p] += tmp[p];
            }
        }
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer6[i], prelu_coeff);
        free(tmp);
    }
}

// ==================== LAYER 7 ====================
void layer7(double *input, double *output, int rows, int cols, int num_threads) {
    int filtersize = 1;
    int padsize = 0;
    int num_filters = 56;
    int num_channels = 12;
    double prelu_coeff = 0.0087;

    omp_set_num_threads(num_threads);
    memset(output, 0, rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++) {
        double *tmp = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < num_channels; j++) {
            imfilter(input + j * rows * cols,
                     weights_layer7 + (i * num_channels + j) * filtersize,
                     tmp, rows, cols, padsize);
            for (int p = 0; p < rows * cols; p++) {
                output[i * rows * cols + p] += tmp[p];
            }
        }
        PReLU(output + i * rows * cols, rows, cols,
              biases_layer7[i], prelu_coeff);
        free(tmp);
    }
}

// ==================== LAYER 8 ====================
void layer8(double *input, double *output, int rows, int cols, int scale, int num_threads) {
    int filtersize = 81; // 9x9
    int num_channels = 56;
    int hr_pixels = (rows * scale) * (cols * scale);

    omp_set_num_threads(num_threads);
    // output: ukuran hr_pixels
    double *accum = (double*)calloc(hr_pixels, sizeof(double));
    if (!accum) return;

    #pragma omp parallel for
    for (int j = 0; j < num_channels; j++) {
        double *tmp = (double*)malloc(hr_pixels * sizeof(double));
        deconv(input + j * rows * cols, tmp,
               weights_layer8 + j * filtersize, cols, rows, scale);
        // imadd ke accum
        for (int p = 0; p < hr_pixels; p++) {
            accum[p] += tmp[p];
        }
        free(tmp);
    }

    // tambah bias
    for (int p = 0; p < hr_pixels; p++) {
        output[p] = accum[p] + biases_layer8;
    }

    free(accum);
}

// ==================== FUNGSI HELPER (dari fsrcnn_parallel.c) ====================

void pad_image(double *img, double *img_pad, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    int i, j, k, cnt, cnt_pad, k1, k2;
    // Central pixels
    for (i = padsize; i < rows_pad - padsize; i++)
    for (j = padsize; j < cols_pad - padsize; j++) {
        cnt_pad = i * cols_pad + j;
        cnt = (i - padsize) * cols + j - padsize;
        *(img_pad + cnt_pad) = *(img + cnt);
    }
    // Top and Bottom Rows
    for (j = padsize; j < cols_pad - padsize; j++)
    for (k = 0; k < padsize; k++) {
        cnt_pad = j + k * cols_pad;
        cnt = j - padsize;
        *(img_pad + cnt_pad) = *(img + cnt);
        cnt_pad = j + (rows_pad - 1 - k) * cols_pad;
        cnt = (j - padsize) + (rows - 1) * cols;
        *(img_pad + cnt_pad) = *(img + cnt);
    }
    // Left and Right Columns
    for (i = padsize; i < rows_pad - padsize; i++)
    for (k = 0; k < padsize; k++) {
        cnt = (i - padsize) * cols;
        cnt_pad = i * cols_pad + k;
        *(img_pad + cnt_pad) = *(img + cnt);
        cnt = (i - padsize) * cols + cols - 1;
        cnt_pad = i * cols_pad + cols_pad - 1 - k;
        *(img_pad + cnt_pad) = *(img + cnt);
    }
    // Corner Pixels
    for (k1 = 0; k1 < padsize; k1++)
    for (k2 = 0; k2 < padsize; k2++) {
        *(img_pad + k1 * cols_pad + k2) = *(img);
        *(img_pad + k1 * cols_pad + cols_pad - 1 - k2) = *(img + cols - 1);
        *(img_pad + (rows_pad - 1 - k1) * cols_pad + k2) = *(img + (rows - 1) * cols);
        *(img_pad + (rows_pad - 1 - k1) * cols_pad + cols_pad - 1 - k2) = *(img + (rows - 1) * cols + cols - 1);
    }
}

void imfilter(double *img, double *kernel, double *img_fltr, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    int i, j, cnt, cnt_pad, cnt_krnl, k1, k2;
    double sum;

    double *img_pad = (double *)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image(img, img_pad, rows, cols, padsize);

    for (i = padsize; i < rows_pad - padsize; i++)
    for (j = padsize; j < cols_pad - padsize; j++) {
        cnt = (i - padsize) * cols + (j - padsize);
        sum = 0;
        cnt_krnl = 0;
        for (k1 = -padsize; k1 <= padsize; k1++)
        for (k2 = -padsize; k2 <= padsize; k2++) {
            cnt_pad = (i + k1) * cols_pad + j + k2;
            sum += (*(img_pad + cnt_pad)) * (*(kernel + cnt_krnl));
            cnt_krnl++;
        }
        *(img_fltr + cnt) = sum;
    }

    free(img_pad);
}

static double _max(double a, double b) { return a > b ? a : b; }
static double _min(double a, double b) { return a > b ? b : a; }

void PReLU(double *img_fltr, int rows, int cols, double bias, double prelu_coeff)
{
    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
        int cnt = i * cols + j;
        *(img_fltr + cnt) = _max(*(img_fltr + cnt) + bias, 0)
                          + prelu_coeff * _min(*(img_fltr + cnt) + bias, 0);
    }
}

void deconv(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride)
{
    int border = 1;
    int fsize  = 9;
    int rows_pad = rows + 2 * border;
    int cols_pad = cols + 2 * border;
    double *img_input_padded = (double *)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image(img_input, img_input_padded, rows, cols, border);

    int rows_out_pad = rows_pad * stride;
    int cols_out_pad = cols_pad * stride;
    double *img_output_tmp = (double *)calloc((rows_out_pad + fsize - 1) * (cols_out_pad + fsize - 1), sizeof(double));
    double *kernel_modif   = (double *)malloc(fsize * fsize * sizeof(double));

    for (int i = 0; i < rows_pad; i++)
    for (int j = 0; j < cols_pad; j++) {
        int cnt_img = i * cols_pad + j;
        int idx = i * stride;
        int idy = j * stride;
        int cnt_img_output = idx * (cols_out_pad + fsize - 1) + idy;
        for (int k_r = 0; k_r < fsize; k_r++) {
            for (int k_c = 0; k_c < fsize; k_c++) {
                int cnt_kernel = k_r * fsize + k_c;
                *(kernel_modif + cnt_kernel) = (*(kernel + cnt_kernel)) * (*(img_input_padded + cnt_img));
                *(img_output_tmp + cnt_img_output + k_c) += *(kernel_modif + cnt_kernel);
            }
            cnt_img_output += (cols_out_pad + fsize - 1);
        }
    }

    int rows_out = rows * stride;
    int cols_out = cols * stride;
    for (int i = 0; i < rows_out; i++)
    for (int j = 0; j < cols_out; j++) {
        int i_tmp = i + ((fsize + 1) / 2) + stride * border - 1;
        int j_tmp = j + ((fsize + 1) / 2) + stride * border - 1;
        int cnt_out     = i * cols_out + j;
        int cnt_out_tmp = i_tmp * (cols_out_pad + fsize - 1) + j_tmp;
        *(img_output + cnt_out) = *(img_output_tmp + cnt_out_tmp);
    }

    free(img_input_padded);
    free(img_output_tmp);
    free(kernel_modif);
}