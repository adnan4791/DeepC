// fsrcnn_layer8_so.c
// Shared library for Layer 8 deconvolution with race condition
// This will be called from Python during training

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Helper functions
void pad_image_deconv(double *img, double *img_pad, int rows, int cols, int padsize);
void deconv_single(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride);
void imadd_race(double *img_fltr_sum, double *img_fltr_crnt, int cols, int rows);

// Global weights for Layer 8 (56 channels, 9x9 kernel each = 56 * 81 = 4536)
static double weights_layer8[4536];
static double bias_layer8 = -0.03262640000;
static int weights_loaded = 0;

// Load weights from file
int load_weights_layer8(const char *filepath)
{
    FILE *fp = fopen(filepath, "r");
    if (fp == NULL)
    {
        printf("Error: Cannot open weights file %s\n", filepath);
        return -1;
    }
    
    for (int i = 0; i < 4536; i++)
    {
        if (fscanf(fp, "%lf", &weights_layer8[i]) != 1)
        {
            printf("Error: Failed to read weight %d\n", i);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    weights_loaded = 1;
    return 0;
}

// Set weights directly from array (for Python training)
void set_weights_layer8(double *new_weights)
{
    memcpy(weights_layer8, new_weights, 4536 * sizeof(double));
    weights_loaded = 1;
}

// Get current weights (for Python to read)
void get_weights_layer8(double *out_weights)
{
    memcpy(out_weights, weights_layer8, 4536 * sizeof(double));
}

// Set bias
void set_bias_layer8(double bias)
{
    bias_layer8 = bias;
}

// Get bias
double get_bias_layer8()
{
    return bias_layer8;
}

// Layer 8 forward pass WITH race condition
// This is the function that produces artifacts
// img_fltr_7: input from layer 7 (rows * cols * 56 doubles)
// img_output: output image (rows*scale * cols*scale doubles)
// rows, cols: input dimensions
// scale: upscaling factor (typically 2)
void layer8_forward_race(double *img_fltr_7, double *img_output, int rows, int cols, int scale)
{
    int filtersize8 = 81; // 9x9
    int num_channels8 = 56;
    
    // Allocate output accumulator
    double *img_fltr_8 = (double *)calloc((rows * scale) * (cols * scale), sizeof(double));
    
    // This is the RACE CONDITION loop - exactly as in original code
    #pragma omp parallel for
    for (int j = 0; j < num_channels8; j++)
    {
        double img_fltr_8_tmp[rows * scale * cols * scale];
        deconv_single(img_fltr_7 + j * rows * cols, img_fltr_8_tmp, 
                      weights_layer8 + j * filtersize8, cols, rows, scale);
        // RACE CONDITION - multiple threads writing to img_fltr_8 simultaneously
        imadd_race(img_fltr_8, img_fltr_8_tmp, cols * scale, rows * scale);
    }
    
    // Add bias and copy to output
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

// Layer 8 forward pass WITHOUT race condition (for comparison)
void layer8_forward_safe(double *img_fltr_7, double *img_output, int rows, int cols, int scale)
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
        
        // Use critical section to prevent race condition
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

// Single channel deconvolution
void deconv_single(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride)
{
    int border = 1;
    int fsize = 9;
    int rows_pad = rows + 2 * border;
    int cols_pad = cols + 2 * border;
    
    double *img_input_padded = (double *)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image_deconv(img_input, img_input_padded, rows, cols, border);
    
    int rows_out_pad = rows_pad * stride;
    int cols_out_pad = cols_pad * stride;
    double *img_output_tmp = (double *)calloc((rows_out_pad + fsize - 1) * (cols_out_pad + fsize - 1), sizeof(double));
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

// Image addition with race condition (no synchronization)
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

// Padding for deconvolution
void pad_image_deconv(double *img, double *img_pad, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    
    // Central pixels
    for (int i = padsize; i < rows_pad - padsize; i++)
    {
        for (int j = padsize; j < cols_pad - padsize; j++)
        {
            int cnt_pad = i * cols_pad + j;
            int cnt = (i - padsize) * cols + j - padsize;
            img_pad[cnt_pad] = img[cnt];
        }
    }
    
    // Top and Bottom Rows
    for (int j = padsize; j < cols_pad - padsize; j++)
    {
        for (int k = 0; k < padsize; k++)
        {
            img_pad[j + k * cols_pad] = img[j - padsize];
            img_pad[j + (rows_pad - 1 - k) * cols_pad] = img[(j - padsize) + (rows - 1) * cols];
        }
    }
    
    // Left and Right Columns
    for (int i = padsize; i < rows_pad - padsize; i++)
    {
        for (int k = 0; k < padsize; k++)
        {
            img_pad[i * cols_pad + k] = img[(i - padsize) * cols];
            img_pad[i * cols_pad + cols_pad - 1 - k] = img[(i - padsize) * cols + cols - 1];
        }
    }
    
    // Corner Pixels
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

// Set number of OpenMP threads
void set_num_threads(int num_threads)
{
    omp_set_num_threads(num_threads);
}

// Get max threads
int get_max_threads()
{
    return omp_get_max_threads();
}
