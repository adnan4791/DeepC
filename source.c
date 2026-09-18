// This code is the C implementation of FSRCNN algorithm for YUV 4:2:0 video interpolation
// Milad Abdollahzadeh, 09/02/2017

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void FSRCNN(double *img_hr, double *img_lr, int rows, int cols, int scale);
// Feature maps between layers are stored HWC-interleaved: pixel (row,col) channel c lives at
// ((row*cols+col)*channels + c), so a pixel's channels are contiguous. Convolution loops over
// pixels first (parallelized), channels innermost -- instead of the old channel-outer order.
void pad_image_hwc(double *img, double *img_pad, int rows, int cols, int channels, int padsize);
void conv_layer_hwc(double *in, int in_channels, double *weights, double *biases, double prelu_coeff,
	double *out, int out_channels, int rows, int cols, int patchsize);
void pad_image(double *img, double *img_pad, int rows, int cols, int padsize);
double Max(double a, double b);
double Min(double a, double b);
void deconv(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride);
void double_2_uint8(double *double_img, unsigned char *uint8_img, int cols, int rows);

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
double weights_layer8[4536];
double biases_layer8 = - 0.03262640000;

int main(int argc, char *argv[])
{
	char *inFile = argv[1];
	char *outFile = argv[2];

	//Upsampler parameters
	int scale = 2;

	//Compressed Assault Cube
	int num = 150; //Number of frames to interpolate
	int inCols = 176; //Width of input (downsampled) video
	int inRows = 144; //Height of input (downsampled) video

	int outCols = inCols*scale;
	int outRows = inRows*scale;

	FILE *inFp, *outFp;

	inFp = fopen(inFile, "rb");
	if (inFp == NULL)
	{
		printf("\n We have null pointer \n");
	}
	outFp = fopen(outFile, "wb");
	if (outFp == NULL)
	{
		printf("\n We have null pointer \n");
	}

    FILE *weights_layer1_ptr;
	weights_layer1_ptr = fopen("weights_layer1.txt", "r");
	if (weights_layer1_ptr == NULL) { printf("Error in the reading weights of first layer\n"); };
	
	for (int i = 0; i < 1400; i++)
	{
		fscanf(weights_layer1_ptr, "%lf", &weights_layer1[i]);
		//printf("%lf\n", weights_layer1[i]);
	}
	fclose(weights_layer1_ptr);
    FILE *biases_layer1_ptr;
	biases_layer1_ptr = fopen("biasess_layer1.txt", "r");
	if (biases_layer1_ptr == NULL) { printf("Error in the reading biases of first layer\n"); };
	
	for (int i = 0; i < 56; i++)
	{
		fscanf(biases_layer1_ptr, "%lf", &biases_layer1[i]);
	}
	fclose(biases_layer1_ptr);
	FILE *weights_layer2_ptr;
    weights_layer2_ptr = fopen("weights_layer2.txt", "r");
	if (weights_layer2_ptr == NULL) { printf("Error in the reading weights of 2nd layer\n"); };
	// Note: weights must be saved in a way which that corresponding weights of each channel can be read by pointer concept ==>> for this layer 12X56 matrix is reshaped to (12X56)*1 vector
	
	for (int i = 0; i < 672; i++)
	{
		fscanf(weights_layer2_ptr, "%lf", &weights_layer2[i]);
	}
	fclose(weights_layer2_ptr);    
    FILE *biases_layer2_ptr;
	biases_layer2_ptr = fopen("biasess_layer2.txt", "r");
	if (biases_layer2_ptr == NULL) { printf("Error in the reading biases of 2nd layer\n"); };
	
	for (int i = 0; i < 12; i++)
	{
		fscanf(biases_layer2_ptr, "%lf", &biases_layer2[i]);
	}
	fclose(biases_layer2_ptr);
    FILE *weights_layer3_ptr;
	weights_layer3_ptr = fopen("weights_layer3.txt", "r");
	if (weights_layer3_ptr == NULL) { printf("Error in the reading weights of 3rd layer\n"); };
	
	for (int i = 0; i < 1296; i++)
	{
		fscanf(weights_layer3_ptr, "%lf", &weights_layer3[i]);
	}
	fclose(weights_layer3_ptr);

	FILE *biases_layer3_ptr;
	biases_layer3_ptr = fopen("biasess_layer3.txt", "r");
	if (biases_layer3_ptr == NULL) { printf("Error in the reading biases of 3rd layer\n"); };
	
	for (int i = 0; i < 12; i++)
	{
		fscanf(biases_layer3_ptr, "%lf", &biases_layer3[i]);
	}
	fclose(biases_layer3_ptr);

    FILE *weights_layer4_ptr;
	weights_layer4_ptr = fopen("weights_layer4.txt", "r");
	if (weights_layer4_ptr == NULL) { printf("Error in the reading weights of 4th layer\n"); };
	
	for (int i = 0; i < 1296; i++)
	{
		fscanf(weights_layer4_ptr, "%lf", &weights_layer4[i]);
	}
	fclose(weights_layer4_ptr);

    FILE *biases_layer4_ptr;
	biases_layer4_ptr = fopen("biasess_layer4.txt", "r");
	if (biases_layer4_ptr == NULL) { printf("Error in the reading biases of 4th layer\n"); };
	
	for (int i = 0; i < 12; i++)
	{
		fscanf(biases_layer4_ptr, "%lf", &biases_layer4[i]);
	}
	fclose(biases_layer4_ptr);

    FILE *weights_layer5_ptr;
	weights_layer5_ptr = fopen("weights_layer5.txt", "r");
	
	for (int i = 0; i < 1296; i++)
	{
		fscanf(weights_layer5_ptr, "%lf", &weights_layer5[i]);
	}
	fclose(weights_layer5_ptr);

	FILE *biases_layer5_ptr;
	biases_layer5_ptr = fopen("biasess_layer5.txt", "r");
	if (biases_layer5_ptr == NULL) { printf("Error in the reading biases of 5th layer\n"); };
	
	for (int i = 0; i < 12; i++)
	{
		fscanf(biases_layer5_ptr, "%lf", &biases_layer5[i]);
	}
	fclose(biases_layer5_ptr);
    
	FILE *weights_layer6_ptr;
	weights_layer6_ptr = fopen("weights_layer6.txt", "r");
	if (weights_layer6_ptr == NULL) { printf("Error in the reading weights of 6th layer\n"); };
	
	for (int i = 0; i < 1296; i++)
	{
		fscanf(weights_layer6_ptr, "%lf", &weights_layer6[i]);
	}
	fclose(weights_layer6_ptr);
	// Reading biases of 6th layer
	FILE *biases_layer6_ptr;
	biases_layer6_ptr = fopen("biasess_layer6.txt", "r");
	if (biases_layer6_ptr == NULL) { printf("Error in the reading biases of 6th layer\n"); };
	for (int i = 0; i < 12; i++)
	{
		fscanf(biases_layer6_ptr, "%lf", &biases_layer6[i]);
	}
	fclose(biases_layer6_ptr);

	FILE *weights_layer7_ptr;
    weights_layer7_ptr = fopen("weights_layer7.txt", "r");
	if (weights_layer7_ptr == NULL) { printf("Error in the reading weights of 7th layer\n"); };
	
	for (int i = 0; i < 672; i++)
	{
		fscanf(weights_layer7_ptr, "%lf", &weights_layer7[i]);
	}
	fclose(weights_layer7_ptr);
	// Reading biases of 7th layer
	FILE *biases_layer7_ptr;
	biases_layer7_ptr = fopen("biasess_layer7.txt", "r");
	if (biases_layer7_ptr == NULL) { printf("Error in the reading biases of 7th layer\n"); };
	
	for (int i = 0; i < 56; i++)
	{
		fscanf(biases_layer7_ptr, "%lf", &biases_layer7[i]);
	}
	fclose(biases_layer7_ptr);

    FILE *weights_layer8_ptr;
	weights_layer8_ptr = fopen("weights_layer8.txt", "r");
	if (weights_layer8_ptr == NULL) { printf("Error in the reading weights of 8th layer\n"); };
	
	for (int i = 0; i < 4536; i++)
	{
		fscanf(weights_layer8_ptr, "%lf", &weights_layer8[i]);
	}
	fclose(weights_layer8_ptr);


	// To read and write each frame in an unsigned character format
	unsigned char *inBuf = (unsigned char *)malloc(inCols*inRows*sizeof(unsigned char));
	unsigned char *outBuf = (unsigned char *)malloc(outCols*outRows*sizeof(unsigned char));
	// To work with each pixel in the range of 0~1
	double *inBuf_tmp = (double *)malloc(inCols*inRows*sizeof(double));
	double *outBuf_tmp = (double *)malloc(outCols*outRows*sizeof(double));

	for (int fcnt = 0; fcnt < num; fcnt++)
	{
		//////// Interpolate each frame using FSRCNN for Y component and simple repitition for U and V components
		// Pointer to obtain value of each tpixel of input frame
		unsigned char *inP = inBuf;
		double *inP_tmp = inBuf_tmp;
		// Pointer to obtain value of each pixel of output frame
		unsigned char *outP = outBuf;
		double *outP_tmp = outBuf_tmp;

		//Y Component
		fread(inBuf, sizeof(unsigned char), inCols*inRows, inFp);
		int i, j;

		for (i = 0; i<inRows; i++)
		for (j = 0; j<inCols; j++)
		{
			int cnt = i*inCols + j;
			int x = *inP++;
			*(inP_tmp + cnt) = (double)(x / 255.0);
		}

		FSRCNN(outP_tmp, inP_tmp, inRows, inCols, scale);

		outP_tmp = outBuf_tmp;
		
		for (i = 0; i<inRows*scale; i++)
		for (j = 0; j<inCols*scale; j++)
		{
			int cnt = i*inCols*scale + j;
			*(outP_tmp + cnt) = *(outP_tmp + cnt) * 255;
		}


		double_2_uint8( outP_tmp, outP, outCols, outRows);

		fwrite(outBuf, sizeof(unsigned char), outCols*outRows, outFp);

		//U Component
		fread(inBuf, sizeof(unsigned char), inCols*inRows / 4, inFp);

		inP = inBuf;
		outP = outBuf;

		for (i = 0; i < inRows / 2; i++)
		for (j = 0; j < inCols / 2; j++) {

			int cnt = 2 * (i * outCols / 2 + j);

			unsigned char x = *inP++;

			*(outP + cnt) = x;
			*(outP + cnt + 1) = x;
			*(outP + cnt + outCols / 2) = x;
			*(outP + cnt + outCols / 2 + 1) = x;

		}

		fwrite(outBuf, sizeof(unsigned char), outCols*outRows / 4, outFp);

		// V COmponent
		fread(inBuf, sizeof(unsigned char), inCols*inRows / 4, inFp);
		inP = inBuf;
		outP= outBuf;

		for (i = 0; i < inRows / 2; i++)
		for (j = 0; j < inCols / 2; j++) {

			int cnt = 2 * (i*outCols / 2 + j);

			unsigned char x = *inP++;

			*(outP + cnt) = x;
			*(outP + cnt + 1) = x;
			*(outP + cnt + outCols / 2) = x;
			*(outP + cnt + outCols / 2 + 1) = x;

		}

		fwrite(outBuf, sizeof(unsigned char), outCols*outRows / 4, outFp);
		
	}
	free(inBuf);
	inBuf = NULL;
	free(inBuf_tmp);
	inBuf_tmp = NULL;
	free(outBuf);
	outBuf = NULL;
	free(outBuf_tmp);
	outBuf_tmp = NULL;
}

void FSRCNN(double *img_hr, double *img_lr, int rows, int cols, int scale)
{
	// General Settings
	int num_layers = 8;

	/////////// Convolution1 -------- Layer1 (5x5, 1 -> 56 channels)
	int patchsize1 = 5;
	int num_filters1 = 56;
	double prelu_coeff_layer1 = -0.8986;
	double *img_fltr_1 = (double *)malloc((size_t)rows * cols * num_filters1 * sizeof(double));
	conv_layer_hwc(img_lr, 1, weights_layer1, biases_layer1, prelu_coeff_layer1,
		img_fltr_1, num_filters1, rows, cols, patchsize1);

	/////////// Convolution2 ------------------- Layer 2~7

	/////////// Layer2 (1x1, 56 -> 12 channels)
	int patchsize2 = 1;
	int num_filters2 = 12;
	int num_channels2 = 56;
	double prelu_coeff_layer2 = 0.3236;
	double *img_fltr_2 = (double *)malloc((size_t)rows * cols * num_filters2 * sizeof(double));
	conv_layer_hwc(img_fltr_1, num_channels2, weights_layer2, biases_layer2, prelu_coeff_layer2,
		img_fltr_2, num_filters2, rows, cols, patchsize2);

	free(img_fltr_1);
	img_fltr_1 = NULL;

	/////////// Layer3 (3x3, 12 -> 12 channels)
	int patchsize3 = 3;
	int num_filters3 = 12;
	int num_channels3 = 12;
	double prelu_coeff_layer3 = 0.2288;
	double *img_fltr_3 = (double *)malloc((size_t)rows * cols * num_filters3 * sizeof(double));
	conv_layer_hwc(img_fltr_2, num_channels3, weights_layer3, biases_layer3, prelu_coeff_layer3,
		img_fltr_3, num_filters3, rows, cols, patchsize3);

	free(img_fltr_2);
	img_fltr_2 = NULL;

	/////////// Layer4 (3x3, 12 -> 12 channels)
	int patchsize4 = 3;
	int num_filters4 = 12;
	int num_channels4 = 12;
	double prelu_coeff_layer4 = 0.2476;
	double *img_fltr_4 = (double *)malloc((size_t)rows * cols * num_filters4 * sizeof(double));
	conv_layer_hwc(img_fltr_3, num_channels4, weights_layer4, biases_layer4, prelu_coeff_layer4,
		img_fltr_4, num_filters4, rows, cols, patchsize4);

	free(img_fltr_3);
	img_fltr_3 = NULL;

	/////////// Layer5 (3x3, 12 -> 12 channels)
	int patchsize5 = 3;
	int num_filters5 = 12;
	int num_channels5 = 12;
	double prelu_coeff_layer5 = 0.3495;
	double *img_fltr_5 = (double *)malloc((size_t)rows * cols * num_filters5 * sizeof(double));
	conv_layer_hwc(img_fltr_4, num_channels5, weights_layer5, biases_layer5, prelu_coeff_layer5,
		img_fltr_5, num_filters5, rows, cols, patchsize5);

	free(img_fltr_4);
	img_fltr_4 = NULL;

	/////////// Layer6 (3x3, 12 -> 12 channels)
	int patchsize6 = 3;
	int num_filters6 = 12;
	int num_channels6 = 12;
	double prelu_coeff_layer6 = 0.7806;
	double *img_fltr_6 = (double *)malloc((size_t)rows * cols * num_filters6 * sizeof(double));
	conv_layer_hwc(img_fltr_5, num_channels6, weights_layer6, biases_layer6, prelu_coeff_layer6,
		img_fltr_6, num_filters6, rows, cols, patchsize6);

	free(img_fltr_5);
	img_fltr_5 = NULL;

	/////////// Layer7 (1x1, 12 -> 56 channels)
	int patchsize7 = 1;
	int num_filters7 = 56;
	int num_channels7 = 12;
	double prelu_coeff_layer7 = 0.0087;
	double *img_fltr_7 = (double *)malloc((size_t)rows * cols * num_filters7 * sizeof(double));
	conv_layer_hwc(img_fltr_6, num_channels7, weights_layer7, biases_layer7, prelu_coeff_layer7,
		img_fltr_7, num_filters7, rows, cols, patchsize7);

	free(img_fltr_6);
	img_fltr_6 = NULL;

	/////////// Convolution3 ------------------- Layer 8 (deconvolution, 9x9, 56 -> 1 channel)
	// Pre-existing in this codebase: this deconvolution (which should populate img_hr from
	// img_fltr_7) was already left unimplemented/commented-out before this HWC refactor.
	int filtersize8 = 81; //9x9
	int patchsize8 = 9;
	int num_filters8 = 1;
	int num_channels8 = 56;

	double *img_fltr_8 = (double *)calloc((rows*scale) *(cols*scale) * num_filters8 , sizeof(double));
	double *kernel8 = (double *)malloc(filtersize8*sizeof(double));

	free(img_fltr_7);
	img_fltr_7 = NULL;

	free(img_fltr_8);
	img_fltr_8 = NULL;
	free(kernel8);
	kernel8 = NULL;

}


// Pad an HWC-interleaved image (rows x cols x channels) with replicate borders into
// (rows+2*padsize) x (cols+2*padsize) x channels, keeping each pixel's channels contiguous.
void pad_image_hwc(double *img, double *img_pad, int rows, int cols, int channels, int padsize)
{
	int cols_pad = cols + 2 * padsize;
	int rows_pad = rows + 2 * padsize;
	int i, j, k, c, k1, k2;

	// Central pixels
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		double *src = img + (size_t)(i*cols + j)*channels;
		double *dst = img_pad + (size_t)((i + padsize)*cols_pad + (j + padsize))*channels;
		for (c = 0; c < channels; c++) dst[c] = src[c];
	}
	// Top and Bottom Rows
	for (j = 0; j < cols; j++)
	for (k = 0; k < padsize; k++)
	{
		double *dst_top = img_pad + (size_t)(k*cols_pad + (j + padsize))*channels;
		double *src_top = img + (size_t)j*channels;
		double *dst_bot = img_pad + (size_t)((rows_pad - 1 - k)*cols_pad + (j + padsize))*channels;
		double *src_bot = img + (size_t)((rows - 1)*cols + j)*channels;
		for (c = 0; c < channels; c++) { dst_top[c] = src_top[c]; dst_bot[c] = src_bot[c]; }
	}
	// Left and Right Columns
	for (i = 0; i < rows; i++)
	for (k = 0; k < padsize; k++)
	{
		double *dst_left = img_pad + (size_t)((i + padsize)*cols_pad + k)*channels;
		double *src_left = img + (size_t)(i*cols)*channels;
		double *dst_right = img_pad + (size_t)((i + padsize)*cols_pad + (cols_pad - 1 - k))*channels;
		double *src_right = img + (size_t)(i*cols + cols - 1)*channels;
		for (c = 0; c < channels; c++) { dst_left[c] = src_left[c]; dst_right[c] = src_right[c]; }
	}
	// Corner Pixels
	for (k1 = 0; k1 < padsize; k1++)
	for (k2 = 0; k2 < padsize; k2++)
	{
		double *ul = img_pad + (size_t)(k1*cols_pad + k2)*channels;
		double *ur = img_pad + (size_t)(k1*cols_pad + cols_pad - 1 - k2)*channels;
		double *ll = img_pad + (size_t)((rows_pad - 1 - k1)*cols_pad + k2)*channels;
		double *lr = img_pad + (size_t)((rows_pad - 1 - k1)*cols_pad + cols_pad - 1 - k2)*channels;
		double *img_tl = img; // (0,0)
		double *img_tr = img + (size_t)(cols - 1)*channels; // (0,cols-1)
		double *img_bl = img + (size_t)((rows - 1)*cols)*channels; // (rows-1,0)
		double *img_br = img + (size_t)((rows - 1)*cols + cols - 1)*channels; // (rows-1,cols-1)
		for (c = 0; c < channels; c++)
		{
			ul[c] = img_tl[c];
			ur[c] = img_tr[c];
			ll[c] = img_bl[c];
			lr[c] = img_br[c];
		}
	}
}

// Convolution + bias + PReLU for one FSRCNN layer, HWC in, HWC out.
// Loop order is pixel-row, pixel-col, output-channel, kernel-row, kernel-col, input-channel --
// height/width outermost (and parallelized over pixels), channel innermost -- matching HWC
// storage: for a fixed output pixel and kernel tap, the input-channel reduction reads contiguous
// memory. Parallelizing over rows*cols pixels (instead of the old per-channel loop) also gives
// far more parallel work than the 12-56 output channels of layers 2-7 alone.
void conv_layer_hwc(double *in, int in_channels, double *weights, double *biases, double prelu_coeff,
	double *out, int out_channels, int rows, int cols, int patchsize)
{
	int padsize = (patchsize - 1) / 2;
	int cols_pad = cols + 2 * padsize;
	int filtersize = patchsize * patchsize;

	double *in_pad = in;
	if (padsize > 0)
	{
		in_pad = (double *)malloc((size_t)(rows + 2 * padsize) * cols_pad * in_channels * sizeof(double));
		pad_image_hwc(in, in_pad, rows, cols, in_channels, padsize);
	}

	// Row-pair blocking: one parallel iteration produces two output rows (oi, oi+1) at once.
	// Each kernel weight w_oc[...] is loaded once and reused for both rows' accumulators,
	// halving the number of OpenMP loop iterations (coarser granularity, less scheduling
	// overhead) and halving redundant weight reads along the row dimension. Odd leftover
	// row (when rows is odd) falls back to a single-row pass.
	#pragma omp parallel for schedule(static)
	for (int oi = 0; oi < rows; oi += 2)
	{
		int two_rows = (oi + 1 < rows);
		for (int oj = 0; oj < cols; oj++)
		{
			size_t out_base0 = (size_t)(oi*cols + oj) * out_channels;
			size_t out_base1 = (size_t)((oi + 1)*cols + oj) * out_channels;
			for (int oc = 0; oc < out_channels; oc++)
			{
				double sum0 = biases[oc];
				double sum1 = biases[oc];
				double *w_oc = weights + (size_t)oc * in_channels * filtersize;
				for (int kr = 0; kr < patchsize; kr++)
				{
					int pr0 = oi + kr;
					int pr1 = oi + 1 + kr;
					for (int kc = 0; kc < patchsize; kc++)
					{
						int pc = oj + kc;
						int k_idx = kr * patchsize + kc;
						double *in_px0 = in_pad + (size_t)(pr0*cols_pad + pc) * in_channels;
						double *in_px1 = in_pad + (size_t)(pr1*cols_pad + pc) * in_channels;
						for (int ic = 0; ic < in_channels; ic++)
						{
							double w = w_oc[ic*filtersize + k_idx];
							sum0 += in_px0[ic] * w;
							if (two_rows) sum1 += in_px1[ic] * w;
						}
					}
				}
				out[out_base0 + oc] = Max(sum0, 0) + prelu_coeff * Min(sum0, 0);
				if (two_rows) out[out_base1 + oc] = Max(sum1, 0) + prelu_coeff * Min(sum1, 0);
			}
		}
	}

	if (padsize > 0)
	{
		free(in_pad);
	}
}


// Replicate image padding by the factor of "padsize"
void pad_image(double *img, double *img_pad, int rows, int cols, int padsize)
{ // This function receives an image and paddes its border in a replicative manner
	int cols_pad = cols + 2 * padsize;
	int rows_pad = rows + 2 * padsize;
	int i, j, k, cnt, cnt_pad, k1, k2;
	// Centeral pixels
	for (i = padsize; i < rows_pad - padsize; i++)
	for (j = padsize; j < cols_pad - padsize; j++)
	{
		cnt_pad = i * cols_pad + j;
		cnt = (i - padsize)*(cols) + j - padsize;
		double x = *(img + cnt);
		*(img_pad + cnt_pad) = x;
	}
	// Top and Bottom Rows
	for (j = padsize; j < cols_pad - padsize; j++)
	for (k = 0; k < padsize; k++)
	{
		// Top Rows 
		cnt_pad = j + k*cols_pad;
		cnt = j - padsize;
		*(img_pad + cnt_pad) = *(img + cnt);
		// Bottom Rows
		cnt_pad = j + (rows_pad - 1 - k)* cols_pad;
		cnt = (j - padsize) + (rows - 1)*cols;
		*(img_pad + cnt_pad) = *(img + cnt);
	}
	// Left and Right Columns
	for (i = padsize; i < rows_pad - padsize; i++)
	for (k = 0; k < padsize; k++)
	{
		// Left Columns
		cnt = (i - padsize)*cols;
		cnt_pad = i*cols_pad + k;
		*(img_pad + cnt_pad) = *(img + cnt);
		// Right Columns
		cnt = (i - padsize)*cols + cols - 1;
		cnt_pad = i*cols_pad + cols_pad - 1 - k;
		*(img_pad + cnt_pad) = *(img + cnt);
	}
	// Corner Pixels
	for (k1 = 0; k1 < padsize; k1++)
	for (k2 = 0; k2 < padsize; k2++)
	{
		// Upper Left Corner
		cnt_pad = k1*cols_pad + k2;
		*(img_pad + cnt_pad) = *(img);
		// Upper Right Corner
		cnt_pad = k1*cols_pad + cols_pad - 1 - k2;
		*(img_pad + cnt_pad) = *(img + cols - 1);
		// Lower Left Corner
		cnt_pad = (rows_pad - 1 - k1)*cols_pad + k2;
		*(img_pad + cnt_pad) = *(img + (rows - 1)*cols);
		// Lower Right Corner
		cnt_pad = (rows_pad - 1 - k1)*cols_pad + cols_pad - 1 - k2;
		*(img_pad + cnt_pad) = *(img + (rows - 1)*cols + cols - 1);
	}
}

double Max(double a, double b)
{
	double c;
	c = a > b ? a : b;
	return c;
}

double Min(double a, double b)
{
	double c;
	c = a > b ? b : a;
	return c;
}


void deconv(double *img_input, double *img_output, double *kernel, int cols, int rows, int stride)
{
	int border = 1;
	int fsize = 9;
	int rows_pad = rows + 2 * border;
	int cols_pad = cols + 2 * border;
	double *img_input_padded = (double *)malloc(rows_pad * cols_pad * sizeof(double));
	pad_image(img_input, img_input_padded, rows, cols, border);
	
	int rows_out_pad = rows_pad * stride;
	int cols_out_pad = cols_pad * stride;
	double *img_output_tmp = (double *)calloc((rows_out_pad + fsize - 1)* (cols_out_pad + fsize - 1), sizeof(double));
	double *kernel_modif = (double *)malloc(fsize * fsize * sizeof(double));

	int idx, idy;
	for (int i = 0; i < rows_pad; i++)
	for (int j = 0; j < cols_pad; j++)
	{
		int cnt_img = i*cols_pad + j;
		idx = i*stride;
		idy = j*stride;
		int cnt_img_output = idx*(cols_out_pad + fsize - 1) + idy; // (idx,idy) coordinate in temporal output image
		int cnt_kernel = 0;
		for (int k_r = 0; k_r < fsize; k_r++)
		{
		for (int k_c = 0; k_c < fsize; k_c++)
		{
			cnt_kernel = k_r*fsize + k_c;
			*(kernel_modif + cnt_kernel) = (*(kernel + cnt_kernel))*(*(img_input_padded + cnt_img));
			*(img_output_tmp + cnt_img_output + k_c) = *(img_output_tmp + cnt_img_output + k_c) + *(kernel_modif + cnt_kernel);
			
		}
		cnt_img_output = cnt_img_output + (cols_out_pad + fsize - 1);
	    }
		
	}

	int rows_out = rows*stride;
	int cols_out = cols*stride;

	for (int i = 0; i < rows_out; i++)
	for (int j = 0; j < cols_out; j++)
	{
		int i_tmp = i + ((fsize + 1) / 2) + stride*border - 1;
		int j_tmp = j + ((fsize + 1) / 2) + stride*border - 1;
		int cnt_img_out = i*cols_out + j;
		int cnt_img_out_tmp = i_tmp*(cols_out_pad + fsize - 1) + j_tmp; // (cols-pad+fsize-1) is the number of columns in the img_out_tmp
		*(img_output + cnt_img_out) = *(img_output_tmp + cnt_img_out_tmp);

	}

	free(img_input_padded); img_input_padded = NULL;
	free(img_output_tmp); img_output_tmp = NULL;
	free(kernel_modif); kernel_modif = NULL;
}

void double_2_uint8(double *double_img, unsigned char *uint8_img, int cols, int rows)
{
	int i, j, cnt, k;

	for (i = 0; i < rows;i++)
	for (j = 0; j < cols; j++)
	{
		cnt = i*cols + j;

		if (*(double_img + cnt) < 0)
			* (uint8_img + cnt) = 0;
		if (*(double_img + cnt) > 255)
			* (uint8_img + cnt) = 255;

		for (k = 0; k < 255; k++)
		{
			if (*(double_img + cnt) >= k && *(double_img + cnt) < (k+0.5))
			*(uint8_img + cnt) =  k;

			if (*(double_img + cnt) >= (k+0.5) && *(double_img + cnt) < (k+1))
				*(uint8_img + cnt) = k + 1;
		}

	}
}
