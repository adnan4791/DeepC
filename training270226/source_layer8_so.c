// source_layer8_so.c
// Shared library for Layer 8 deconvolution with race condition
// ENHANCED: Exposes per-step functions (deconv, imadd, bias) individually
// for proper per-step error backpropagation in Python training.
//
// Compile:
//   gcc -shared -o fsrcnn_layer8_3step.so -fPIC source_layer8_so.c -fopenmp -O2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// ============================================================================
// Helper function declarations
// ============================================================================
void pad_image_deconv(double *img, double *img_pad, int rows, int cols, int padsize);
void deconv_single(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride);
void imadd_race(double *img_fltr_sum, double *img_fltr_crnt, int cols, int rows);

// ============================================================================
// Global state
// ============================================================================
static double weights_layer8[4536];  // 56 channels x 81 (9x9)
static double bias_layer8 = -0.03262640000;
static int weights_loaded = 0;

// ============================================================================
// Weight/Bias accessors (for Python)
// ============================================================================
void set_weights_layer8(double *new_weights)
{
    memcpy(weights_layer8, new_weights, 4536 * sizeof(double));
    weights_loaded = 1;
}

void get_weights_layer8(double *out_weights)
{
    memcpy(out_weights, weights_layer8, 4536 * sizeof(double));
}

void set_bias_layer8(double bias)
{
    bias_layer8 = bias;
}

double get_bias_layer8()
{
    return bias_layer8;
}

// ============================================================================
// STEP 1: Deconvolution - Single channel (EXPORTED)
// ============================================================================
// Performs deconvolution for a single channel with given kernel.
// img_input:  (rows x cols) single channel input
// img_output: (rows*stride x cols*stride) single channel output
// kernel:     (9x9 = 81) kernel weights
// cols, rows: input dimensions
// stride:     upscaling factor
void deconv_single_export(double *img_input, double *img_output, 
                          double *kernel, int cols, int rows, int stride)
{
    deconv_single(img_input, img_output, kernel, cols, rows, stride);
}

// Batch deconvolution: all 56 channels, outputs kept separate (56 planes)
// img_fltr_7:    input from layer 7, shape: 56 * rows * cols (contiguous)
// img_deconv_56: output, shape: 56 * (rows*stride) * (cols*stride) (contiguous)
// Uses current global weights
void deconv_all_channels(double *img_fltr_7, double *img_deconv_56,
                         int rows, int cols, int stride)
{
    int filtersize8 = 81;
    int num_channels8 = 56;
    int out_size = rows * stride * cols * stride;
    
    // Each channel independently (can be parallel since outputs are separate)
    #pragma omp parallel for
    for (int j = 0; j < num_channels8; j++)
    {
        deconv_single(img_fltr_7 + j * rows * cols,
                      img_deconv_56 + j * out_size,
                      weights_layer8 + j * filtersize8,
                      cols, rows, stride);
    }
}

// ============================================================================
// STEP 2: Imadd with Race Condition (EXPORTED)
// ============================================================================
// Sums 56 channels into 1 channel WITH race condition (no synchronization)
// input_56ch:  56 separate deconv outputs, shape: 56 * out_rows * out_cols
// output_1ch:  summed output, shape: out_rows * out_cols
// out_rows, out_cols: output dimensions (rows*scale, cols*scale)
void imadd_race_all_channels(double *input_56ch, double *output_1ch,
                              int out_rows, int out_cols, int num_channels)
{
    int frame_size = out_rows * out_cols;
    
    // Initialize output to zero
    memset(output_1ch, 0, frame_size * sizeof(double));
    
    // OpenMP parallel WITHOUT synchronization = RACE CONDITION
    #pragma omp parallel for
    for (int j = 0; j < num_channels; j++)
    {
        double *current_channel = input_56ch + j * frame_size;
        imadd_race(output_1ch, current_channel, out_cols, out_rows);
    }
}

// ============================================================================
// STEP 2b: Imadd SAFE (for gradient estimation)
// ============================================================================
// Same as above but WITH synchronization (deterministic result)
void imadd_safe_all_channels(double *input_56ch, double *output_1ch,
                              int out_rows, int out_cols, int num_channels)
{
    int frame_size = out_rows * out_cols;
    
    memset(output_1ch, 0, frame_size * sizeof(double));
    
    #pragma omp parallel for
    for (int j = 0; j < num_channels; j++)
    {
        double *current_channel = input_56ch + j * frame_size;
        
        #pragma omp critical
        {
            imadd_race(output_1ch, current_channel, out_cols, out_rows);
        }
    }
}

// ============================================================================
// Combined forward passes (kept for compatibility)
// ============================================================================
void layer8_forward_race(double *img_fltr_7, double *img_output, 
                         int rows, int cols, int scale)
{
    int filtersize8 = 81;
    int num_channels8 = 56;
    
    double *img_fltr_8 = (double *)calloc((rows * scale) * (cols * scale), sizeof(double));
    
    #pragma omp parallel for
    for (int j = 0; j < num_channels8; j++)
    {
        double img_fltr_8_tmp[rows * scale * cols * scale];
        deconv_single(img_fltr_7 + j * rows * cols, img_fltr_8_tmp,
                      weights_layer8 + j * filtersize8, cols, rows, scale);
        imadd_race(img_fltr_8, img_fltr_8_tmp, cols * scale, rows * scale);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < rows * scale; i++)
    {
        for (int j = 0; j < cols * scale; j++)
        {
            int cnt_fnl = i * cols * scale + j;
            img_output[cnt_fnl] = img_fltr_8[cnt_fnl] + bias_layer8;
        }
    }
    
    free(img_fltr_8);
}

void layer8_forward_safe(double *img_fltr_7, double *img_output,
                         int rows, int cols, int scale)
{
    int filtersize8 = 81;
    int num_channels8 = 56;
    
    double *img_fltr_8 = (double *)calloc((rows * scale) * (cols * scale), sizeof(double));
    
    #pragma omp parallel for
    for (int j = 0; j < num_channels8; j++)
    {
        double img_fltr_8_tmp[rows * scale * cols * scale];
        deconv_single(img_fltr_7 + j * rows * cols, img_fltr_8_tmp,
                      weights_layer8 + j * filtersize8, cols, rows, scale);
        
        #pragma omp critical
        {
            imadd_race(img_fltr_8, img_fltr_8_tmp, cols * scale, rows * scale);
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < rows * scale; i++)
    {
        for (int j = 0; j < cols * scale; j++)
        {
            int cnt_fnl = i * cols * scale + j;
            img_output[cnt_fnl] = img_fltr_8[cnt_fnl] + bias_layer8;
        }
    }
    
    free(img_fltr_8);
}

// ============================================================================
// Core implementations
// ============================================================================
void deconv_single(double *img_input, double *img_output, double *kernel,
                   int cols, int rows, int stride)
{
    int border = 1;
    int fsize = 9;
    int rows_pad = rows + 2 * border;
    int cols_pad = cols + 2 * border;
    
    double *img_input_padded = (double *)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image_deconv(img_input, img_input_padded, rows, cols, border);
    
    int rows_out_pad = rows_pad * stride;
    int cols_out_pad = cols_pad * stride;
    double *img_output_tmp = (double *)calloc(
        (rows_out_pad + fsize - 1) * (cols_out_pad + fsize - 1), sizeof(double));
    double *kernel_modif = (double *)malloc(fsize * fsize * sizeof(double));
    
    for (int i = 0; i < rows_pad; i++)
    {
        for (int j = 0; j < cols_pad; j++)
        {
            int cnt_img = i * cols_pad + j;
            int idx = i * stride;
            int idy = j * stride;
            int cnt_img_output = idx * (cols_out_pad + fsize - 1) + idy;
            
            for (int k_r = 0; k_r < fsize; k_r++)
            {
                for (int k_c = 0; k_c < fsize; k_c++)
                {
                    int cnt_kernel = k_r * fsize + k_c;
                    kernel_modif[cnt_kernel] = kernel[cnt_kernel] * img_input_padded[cnt_img];
                    img_output_tmp[cnt_img_output + k_c] += kernel_modif[cnt_kernel];
                }
                cnt_img_output += (cols_out_pad + fsize - 1);
            }
        }
    }
    
    int rows_out = rows * stride;
    int cols_out = cols * stride;
    
    for (int i = 0; i < rows_out; i++)
    {
        for (int j = 0; j < cols_out; j++)
        {
            int i_tmp = i + (fsize + 1) / 2 + stride * border - 1;
            int j_tmp = j + (fsize + 1) / 2 + stride * border - 1;
            int cnt_img_out = i * cols_out + j;
            int cnt_img_out_tmp = i_tmp * (cols_out_pad + fsize - 1) + j_tmp;
            img_output[cnt_img_out] = img_output_tmp[cnt_img_out_tmp];
        }
    }
    
    free(img_input_padded);
    free(img_output_tmp);
    free(kernel_modif);
}

void imadd_race(double *img_fltr_sum, double *img_fltr_crnt, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int cnt = i * cols + j;
            img_fltr_sum[cnt] = img_fltr_sum[cnt] + img_fltr_crnt[cnt];
        }
    }
}

void pad_image_deconv(double *img, double *img_pad, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    
    for (int i = padsize; i < rows_pad - padsize; i++)
    {
        for (int j = padsize; j < cols_pad - padsize; j++)
        {
            int cnt_pad = i * cols_pad + j;
            int cnt = (i - padsize) * cols + j - padsize;
            img_pad[cnt_pad] = img[cnt];
        }
    }
    
    for (int j = padsize; j < cols_pad - padsize; j++)
    {
        for (int k = 0; k < padsize; k++)
        {
            img_pad[j + k * cols_pad] = img[j - padsize];
            img_pad[j + (rows_pad - 1 - k) * cols_pad] = img[(j - padsize) + (rows - 1) * cols];
        }
    }
    
    for (int i = padsize; i < rows_pad - padsize; i++)
    {
        for (int k = 0; k < padsize; k++)
        {
            img_pad[i * cols_pad + k] = img[(i - padsize) * cols];
            img_pad[i * cols_pad + cols_pad - 1 - k] = img[(i - padsize) * cols + cols - 1];
        }
    }
    
    for (int k1 = 0; k1 < padsize; k1++)
    {
        for (int k2 = 0; k2 < padsize; k2++)
        {
            img_pad[k1 * cols_pad + k2] = img[0];
            img_pad[k1 * cols_pad + cols_pad - 1 - k2] = img[cols - 1];
            img_pad[(rows_pad - 1 - k1) * cols_pad + k2] = img[(rows - 1) * cols];
            img_pad[(rows_pad - 1 - k1) * cols_pad + cols_pad - 1 - k2] = img[(rows - 1) * cols + cols - 1];
        }
    }
}

// ============================================================================
// Thread control
// ============================================================================
void set_num_threads(int num_threads)
{
    omp_set_num_threads(num_threads);
}

int get_max_threads()
{
    return omp_get_max_threads();
}
