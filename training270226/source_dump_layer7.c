// This code dumps Layer 7 output for retraining Layer 8
// Modified from source_4.c to export intermediate results

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

void FSRCNN_dump_layer7(double *img_lr, int rows, int cols, int scale, const char *layer7_file, const char *hr_file, double *ground_truth_hr);
void imfilter(double *img, double *kernel, double *img_fltr, int rows, int cols, int padsize);
void pad_image(double *img, double *img_pad, int rows, int cols, int padsize);
void PReLU(double *img_fltr, int rows, int cols, double bias, double prelu_coeff);
double Max(double a, double b);
double Min(double a, double b);
void imadd(double *img_fltr_crnt, double *img_fltr_prev, int cols, int rows);

// Global weights
double weights_layer1[1400];
double biases_layer1[56];
double weights_layer2[672];
double biases_layer2[12];
double weights_layer3[1296];
double biases_layer3[12];
double weights_layer4[1296];
double biases_layer4[12];
double weights_layer5[1296];
double biases_layer5[12];
double weights_layer6[1296];
double biases_layer6[12];
double weights_layer7[672];
double biases_layer7[56];

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <input_lr.yuv> <hr_serial.yuv> <num_frames>\n", argv[0]);
        return 1;
    }

    char *inFile = argv[1];
    char *hrFile = argv[2];
    int num = atoi(argv[3]);

    // Upsampler parameters
    int scale = 2;

    // Video dimensions
    int inCols = 176;
    int inRows = 144;

    int outCols = inCols * scale;
    int outRows = inRows * scale;

    FILE *inFp;

    inFp = fopen(inFile, "rb");
    if (inFp == NULL)
    {
        printf("Error: Cannot open input file %s\n", inFile);
        return 1;
    }

    // Open serial ground truth YUV file
    FILE *serialFp = fopen(hrFile, "rb");
    if (serialFp == NULL)
    {
        printf("Error: Cannot open HR serial file %s\n", hrFile);
        fclose(inFp);
        return 1;
    }

    // Load weights
    FILE *weights_layer1_ptr = fopen("weights_layer1.txt", "r");
    if (weights_layer1_ptr == NULL) { printf("Error reading weights_layer1.txt\n"); return 1; }
    for (int i = 0; i < 1400; i++) fscanf(weights_layer1_ptr, "%lf", &weights_layer1[i]);
    fclose(weights_layer1_ptr);

    FILE *biases_layer1_ptr = fopen("biasess_layer1.txt", "r");
    if (biases_layer1_ptr == NULL) { printf("Error reading biasess_layer1.txt\n"); return 1; }
    for (int i = 0; i < 56; i++) fscanf(biases_layer1_ptr, "%lf", &biases_layer1[i]);
    fclose(biases_layer1_ptr);

    FILE *weights_layer2_ptr = fopen("weights_layer2.txt", "r");
    if (weights_layer2_ptr == NULL) { printf("Error reading weights_layer2.txt\n"); return 1; }
    for (int i = 0; i < 672; i++) fscanf(weights_layer2_ptr, "%lf", &weights_layer2[i]);
    fclose(weights_layer2_ptr);

    FILE *biases_layer2_ptr = fopen("biasess_layer2.txt", "r");
    if (biases_layer2_ptr == NULL) { printf("Error reading biasess_layer2.txt\n"); return 1; }
    for (int i = 0; i < 12; i++) fscanf(biases_layer2_ptr, "%lf", &biases_layer2[i]);
    fclose(biases_layer2_ptr);

    FILE *weights_layer3_ptr = fopen("weights_layer3.txt", "r");
    if (weights_layer3_ptr == NULL) { printf("Error reading weights_layer3.txt\n"); return 1; }
    for (int i = 0; i < 1296; i++) fscanf(weights_layer3_ptr, "%lf", &weights_layer3[i]);
    fclose(weights_layer3_ptr);

    FILE *biases_layer3_ptr = fopen("biasess_layer3.txt", "r");
    if (biases_layer3_ptr == NULL) { printf("Error reading biasess_layer3.txt\n"); return 1; }
    for (int i = 0; i < 12; i++) fscanf(biases_layer3_ptr, "%lf", &biases_layer3[i]);
    fclose(biases_layer3_ptr);

    FILE *weights_layer4_ptr = fopen("weights_layer4.txt", "r");
    if (weights_layer4_ptr == NULL) { printf("Error reading weights_layer4.txt\n"); return 1; }
    for (int i = 0; i < 1296; i++) fscanf(weights_layer4_ptr, "%lf", &weights_layer4[i]);
    fclose(weights_layer4_ptr);

    FILE *biases_layer4_ptr = fopen("biasess_layer4.txt", "r");
    if (biases_layer4_ptr == NULL) { printf("Error reading biasess_layer4.txt\n"); return 1; }
    for (int i = 0; i < 12; i++) fscanf(biases_layer4_ptr, "%lf", &biases_layer4[i]);
    fclose(biases_layer4_ptr);

    FILE *weights_layer5_ptr = fopen("weights_layer5.txt", "r");
    if (weights_layer5_ptr == NULL) { printf("Error reading weights_layer5.txt\n"); return 1; }
    for (int i = 0; i < 1296; i++) fscanf(weights_layer5_ptr, "%lf", &weights_layer5[i]);
    fclose(weights_layer5_ptr);

    FILE *biases_layer5_ptr = fopen("biasess_layer5.txt", "r");
    if (biases_layer5_ptr == NULL) { printf("Error reading biasess_layer5.txt\n"); return 1; }
    for (int i = 0; i < 12; i++) fscanf(biases_layer5_ptr, "%lf", &biases_layer5[i]);
    fclose(biases_layer5_ptr);

    FILE *weights_layer6_ptr = fopen("weights_layer6.txt", "r");
    if (weights_layer6_ptr == NULL) { printf("Error reading weights_layer6.txt\n"); return 1; }
    for (int i = 0; i < 1296; i++) fscanf(weights_layer6_ptr, "%lf", &weights_layer6[i]);
    fclose(weights_layer6_ptr);

    FILE *biases_layer6_ptr = fopen("biasess_layer6.txt", "r");
    if (biases_layer6_ptr == NULL) { printf("Error reading biasess_layer6.txt\n"); return 1; }
    for (int i = 0; i < 12; i++) fscanf(biases_layer6_ptr, "%lf", &biases_layer6[i]);
    fclose(biases_layer6_ptr);

    FILE *weights_layer7_ptr = fopen("weights_layer7.txt", "r");
    if (weights_layer7_ptr == NULL) { printf("Error reading weights_layer7.txt\n"); return 1; }
    for (int i = 0; i < 672; i++) fscanf(weights_layer7_ptr, "%lf", &weights_layer7[i]);
    fclose(weights_layer7_ptr);

    FILE *biases_layer7_ptr = fopen("biasess_layer7.txt", "r");
    if (biases_layer7_ptr == NULL) { printf("Error reading biasess_layer7.txt\n"); return 1; }
    for (int i = 0; i < 56; i++) fscanf(biases_layer7_ptr, "%lf", &biases_layer7[i]);
    fclose(biases_layer7_ptr);

    // Buffers
    unsigned char *inBuf = (unsigned char *)malloc(inCols * inRows * sizeof(unsigned char));
    double *inBuf_tmp = (double *)malloc(inCols * inRows * sizeof(double));

    // Buffer for serial ground truth HR frame (Y component at output resolution)
    unsigned char *serialBuf = (unsigned char *)malloc(outCols * outRows * sizeof(unsigned char));
    double *serialBuf_tmp = (double *)malloc(outCols * outRows * sizeof(double));

    // Create directory for training data
    system("mkdir -p training_data");

    for (int fcnt = 0; fcnt < num; fcnt++)
    {
        // Y Component
        fread(inBuf, sizeof(unsigned char), inCols * inRows, inFp);

        for (int i = 0; i < inRows; i++)
            for (int j = 0; j < inCols; j++)
            {
                int cnt = i * inCols + j;
                int x = inBuf[cnt];
                inBuf_tmp[cnt] = (double)(x / 255.0);
            }

        // Create filenames for this frame
        char layer7_file[256], hr_file[256];
        sprintf(layer7_file, "training_data/layer7_frame%04d.bin", fcnt);
        sprintf(hr_file, "training_data/hr_frame%04d.bin", fcnt);

        // Read serial ground truth HR frame (Y component)
        fread(serialBuf, sizeof(unsigned char), outCols * outRows, serialFp);
        for (int i = 0; i < outRows; i++)
            for (int j = 0; j < outCols; j++)
            {
                int cnt = i * outCols + j;
                serialBuf_tmp[cnt] = (double)(serialBuf[cnt] / 255.0);
            }
        // Skip U and V components of serial ground truth
        fseek(serialFp, outCols * outRows / 4, SEEK_CUR); // U
        fseek(serialFp, outCols * outRows / 4, SEEK_CUR); // V

        // Process and dump
        FSRCNN_dump_layer7(inBuf_tmp, inRows, inCols, scale, layer7_file, hr_file, serialBuf_tmp);

        // Skip U and V components in input YUV (Y-only processing for training)
        fseek(inFp, inCols * inRows / 4, SEEK_CUR); // U
        fseek(inFp, inCols * inRows / 4, SEEK_CUR); // V

        printf("Processed frame %d/%d\n", fcnt + 1, num);
    }

    // Save dimensions info
    FILE *dim_file = fopen("training_data/dimensions.txt", "w");
    fprintf(dim_file, "%d %d %d %d\n", inRows, inCols, scale, num);
    fclose(dim_file);

    free(inBuf);
    free(inBuf_tmp);
    free(serialBuf);
    free(serialBuf_tmp);
    fclose(inFp);
    fclose(serialFp);

    printf("Done! Training data saved to training_data/\n");
    return 0;
}

void FSRCNN_dump_layer7(double *img_lr, int rows, int cols, int scale, const char *layer7_file, const char *hr_file, double *ground_truth_hr)
{
    // Layer 1
    int filtersize = 25;
    int padsize = 2;
    int num_filters = 56;
    double prelu_coeff_layer1 = -0.8986;

    double *img_fltr_1 = (double *)malloc(rows * cols * num_filters * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters; i++)
    {
        imfilter(img_lr, weights_layer1 + i * filtersize, img_fltr_1 + i * cols * rows, rows, cols, padsize);
        PReLU(img_fltr_1 + i * cols * rows, rows, cols, biases_layer1[i], prelu_coeff_layer1);
    }

    // Layer 2
    int filtersize2 = 1;
    int padsize2 = 0;
    int num_filters2 = 12;
    int num_channels2 = 56;
    double prelu_coeff_layer2 = 0.3236;

    double *img_fltr_2 = (double *)calloc(rows * cols * num_filters2, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters2; i++)
    {
        double img_fltr_2_tmp[rows * cols];
        for (int j = 0; j < num_channels2; j++)
        {
            imfilter(img_fltr_1 + j * rows * cols, weights_layer2 + (i * num_channels2 + j) * filtersize2, img_fltr_2_tmp, rows, cols, padsize2);
            imadd(img_fltr_2 + i * cols * rows, img_fltr_2_tmp, cols, rows);
        }
        PReLU(img_fltr_2 + i * rows * cols, rows, cols, biases_layer2[i], prelu_coeff_layer2);
    }
    free(img_fltr_1);

    // Layer 3
    int filtersize3 = 9;
    int padsize3 = 1;
    int num_filters3 = 12;
    int num_channels3 = 12;
    double prelu_coeff_layer3 = 0.2288;

    double *img_fltr_3 = (double *)calloc(rows * cols * num_filters3, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters3; i++)
    {
        double img_fltr_3_tmp[rows * cols];
        for (int j = 0; j < num_channels3; j++)
        {
            imfilter(img_fltr_2 + j * rows * cols, weights_layer3 + (i * num_channels3 + j) * filtersize3, img_fltr_3_tmp, rows, cols, padsize3);
            imadd(img_fltr_3 + i * rows * cols, img_fltr_3_tmp, cols, rows);
        }
        PReLU(img_fltr_3 + i * rows * cols, rows, cols, biases_layer3[i], prelu_coeff_layer3);
    }
    free(img_fltr_2);

    // Layer 4
    int filtersize4 = 9;
    int padsize4 = 1;
    int num_filters4 = 12;
    int num_channels4 = 12;
    double prelu_coeff_layer4 = 0.2476;

    double *img_fltr_4 = (double *)calloc(rows * cols * num_filters4, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters4; i++)
    {
        double img_fltr_4_tmp[rows * cols];
        for (int j = 0; j < num_channels4; j++)
        {
            imfilter(img_fltr_3 + j * rows * cols, weights_layer4 + (i * num_channels4 + j) * filtersize4, img_fltr_4_tmp, rows, cols, padsize4);
            imadd(img_fltr_4 + i * rows * cols, img_fltr_4_tmp, cols, rows);
        }
        PReLU(img_fltr_4 + i * rows * cols, rows, cols, biases_layer4[i], prelu_coeff_layer4);
    }
    free(img_fltr_3);

    // Layer 5
    int filtersize5 = 9;
    int padsize5 = 1;
    int num_filters5 = 12;
    int num_channels5 = 12;
    double prelu_coeff_layer5 = 0.3495;

    double *img_fltr_5 = (double *)calloc(rows * cols * num_filters5, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters5; i++)
    {
        double img_fltr_5_tmp[rows * cols];
        for (int j = 0; j < num_channels5; j++)
        {
            imfilter(img_fltr_4 + j * rows * cols, weights_layer5 + (i * num_channels5 + j) * filtersize5, img_fltr_5_tmp, rows, cols, padsize5);
            imadd(img_fltr_5 + i * rows * cols, img_fltr_5_tmp, cols, rows);
        }
        PReLU(img_fltr_5 + i * rows * cols, rows, cols, biases_layer5[i], prelu_coeff_layer5);
    }
    free(img_fltr_4);

    // Layer 6
    int filtersize6 = 9;
    int padsize6 = 1;
    int num_filters6 = 12;
    int num_channels6 = 12;
    double prelu_coeff_layer6 = 0.7806;

    double *img_fltr_6 = (double *)calloc(rows * cols * num_filters6, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters6; i++)
    {
        double img_fltr_6_tmp[rows * cols];
        for (int j = 0; j < num_channels6; j++)
        {
            imfilter(img_fltr_5 + j * rows * cols, weights_layer6 + (i * num_channels6 + j) * filtersize6, img_fltr_6_tmp, rows, cols, padsize6);
            imadd(img_fltr_6 + i * rows * cols, img_fltr_6_tmp, cols, rows);
        }
        PReLU(img_fltr_6 + i * rows * cols, rows, cols, biases_layer6[i], prelu_coeff_layer6);
    }
    free(img_fltr_5);

    // Layer 7
    int filtersize7 = 1;
    int padsize7 = 0;
    int num_filters7 = 56;
    int num_channels7 = 12;
    double prelu_coeff_layer7 = 0.0087;

    double *img_fltr_7 = (double *)calloc(rows * cols * num_filters7, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < num_filters7; i++)
    {
        double img_fltr_7_tmp[rows * cols];
        for (int j = 0; j < num_channels7; j++)
        {
            imfilter(img_fltr_6 + j * rows * cols, weights_layer7 + (i * num_channels7 + j) * filtersize7, img_fltr_7_tmp, rows, cols, padsize7);
            imadd(img_fltr_7 + i * rows * cols, img_fltr_7_tmp, cols, rows);
        }
        PReLU(img_fltr_7 + i * rows * cols, rows, cols, biases_layer7[i], prelu_coeff_layer7);
    }
    free(img_fltr_6);

    // DUMP LAYER 7 OUTPUT
    FILE *fp_l7 = fopen(layer7_file, "wb");
    fwrite(img_fltr_7, sizeof(double), rows * cols * num_filters7, fp_l7);
    fclose(fp_l7);

    // DUMP HR ground truth from serial output (correct, race-condition-free result)
    FILE *fp_hr = fopen(hr_file, "wb");
    fwrite(ground_truth_hr, sizeof(double), rows * scale * cols * scale, fp_hr);
    fclose(fp_hr);

    free(img_fltr_7);
}


// Helper functions (same as original)
void imfilter(double *img, double *kernel, double *img_fltr, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;
    double *img_pad = (double *)malloc(rows_pad * cols_pad * sizeof(double));
    pad_image(img, img_pad, rows, cols, padsize);

    for (int i = padsize; i < rows_pad - padsize; i++)
        for (int j = padsize; j < cols_pad - padsize; j++)
        {
            int cnt = (i - padsize) * cols + (j - padsize);
            double sum = 0;
            int cnt_krnl = 0;
            for (int k1 = -padsize; k1 <= padsize; k1++)
                for (int k2 = -padsize; k2 <= padsize; k2++)
                {
                    int cnt_pad = (i + k1) * cols_pad + j + k2;
                    sum = sum + img_pad[cnt_pad] * kernel[cnt_krnl];
                    cnt_krnl++;
                }
            img_fltr[cnt] = sum;
        }

    free(img_pad);
}

void pad_image(double *img, double *img_pad, int rows, int cols, int padsize)
{
    int cols_pad = cols + 2 * padsize;
    int rows_pad = rows + 2 * padsize;

    // Central pixels
    for (int i = padsize; i < rows_pad - padsize; i++)
        for (int j = padsize; j < cols_pad - padsize; j++)
        {
            int cnt_pad = i * cols_pad + j;
            int cnt = (i - padsize) * cols + j - padsize;
            img_pad[cnt_pad] = img[cnt];
        }

    // Top and Bottom Rows
    for (int j = padsize; j < cols_pad - padsize; j++)
        for (int k = 0; k < padsize; k++)
        {
            img_pad[j + k * cols_pad] = img[j - padsize];
            img_pad[j + (rows_pad - 1 - k) * cols_pad] = img[(j - padsize) + (rows - 1) * cols];
        }

    // Left and Right Columns
    for (int i = padsize; i < rows_pad - padsize; i++)
        for (int k = 0; k < padsize; k++)
        {
            img_pad[i * cols_pad + k] = img[(i - padsize) * cols];
            img_pad[i * cols_pad + cols_pad - 1 - k] = img[(i - padsize) * cols + cols - 1];
        }

    // Corner Pixels
    for (int k1 = 0; k1 < padsize; k1++)
        for (int k2 = 0; k2 < padsize; k2++)
        {
            img_pad[k1 * cols_pad + k2] = img[0];
            img_pad[k1 * cols_pad + cols_pad - 1 - k2] = img[cols - 1];
            img_pad[(rows_pad - 1 - k1) * cols_pad + k2] = img[(rows - 1) * cols];
            img_pad[(rows_pad - 1 - k1) * cols_pad + cols_pad - 1 - k2] = img[(rows - 1) * cols + cols - 1];
        }
}

void PReLU(double *img_fltr, int rows, int cols, double bias, double prelu_coeff)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            int cnt = i * cols + j;
            double val = img_fltr[cnt] + bias;
            img_fltr[cnt] = (val > 0 ? val : 0) + prelu_coeff * (val < 0 ? val : 0);
        }
}

double Max(double a, double b) { return a > b ? a : b; }
double Min(double a, double b) { return a > b ? b : a; }

void imadd(double *img_fltr_sum, double *img_fltr_crnt, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            int cnt = i * cols + j;
            img_fltr_sum[cnt] = img_fltr_sum[cnt] + img_fltr_crnt[cnt];
        }
}
