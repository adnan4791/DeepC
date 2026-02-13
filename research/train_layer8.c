
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FRAMES 50
#define CHANNELS 56
#define ROWS 144
#define COLS 176
#define SCALE 2
#define KERNEL_SIZE 9
#define STRIDE 2
#define TARGET_ROWS (ROWS * SCALE)
#define TARGET_COLS (COLS * SCALE)
#define NUM_ITERATIONS 5000000
#define LR 0.0001
#define DROPOUT_RATE 0.3

// Global buffers
double *inputs; // FRAMES * CHANNELS * ROWS * COLS
double *targets; // FRAMES * TARGET_ROWS * TARGET_COLS
double weights[CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
double bias = 0.0;

// Helper to get random double 0..1
double rand_double() {
    return (double)rand() / (double)RAND_MAX;
}

void load_data() {
    // Load Layer 7 inputs
    FILE *f = fopen("layer7.bin", "rb");
    if (!f) { printf("Error opening layer7.bin\n"); exit(1); }
    
    long input_size = (long)FRAMES * CHANNELS * ROWS * COLS;
    inputs = (double*)malloc(input_size * sizeof(double));
    if (!inputs) { printf("Memory alloc failed for inputs\n"); exit(1); }
    
    size_t read_count = fread(inputs, sizeof(double), input_size, f);
    if (read_count != input_size) { printf("Warning: Read %zu doubles needed %ld\n", read_count, input_size); }
    fclose(f);
    
    // Load Targets (Y channel of CIF YUV)
    f = fopen("akiyo_cif.yuv", "rb");
    if (!f) { printf("Error opening akiyo_cif.yuv\n"); exit(1); }
    
    long target_size = (long)FRAMES * TARGET_ROWS * TARGET_COLS;
    targets = (double*)malloc(target_size * sizeof(double));
    if (!targets) { printf("Memory alloc failed for targets\n"); exit(1); }
    
    int frame_size_y = TARGET_ROWS * TARGET_COLS;
    int frame_size_uv = (TARGET_ROWS/2) * (TARGET_COLS/2);
    int total_frame_bytes = frame_size_y + 2 * frame_size_uv;
    
    unsigned char *buf = (unsigned char*)malloc(frame_size_y);
    
    for (int i = 0; i < FRAMES; i++) {
        // Seek to Y start
        fseek(f, i * total_frame_bytes, SEEK_SET);
        fread(buf, 1, frame_size_y, f);
        
        for (int j = 0; j < frame_size_y; j++) {
            targets[i * frame_size_y + j] = (double)buf[j] / 255.0;
        }
    }
    free(buf);
    fclose(f);
    printf("Data loaded.\n");
}

void init_weights() {
    srand(time(NULL));
    for (int c = 0; c < CHANNELS; c++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                // Initialize small random weights
                weights[c][i][j] = (rand_double() - 0.5) * 0.1;
            }
        }
    }
    bias = 0.0;
}

void save_weights() {
    FILE *f = fopen("weights_layer8_robust.txt", "w");
    if (!f) return;
    
    // Format: Channel 0 (81 values), Channel 1...
    for (int c = 0; c < CHANNELS; c++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                fprintf(f, "%.10f\n", weights[c][i][j]);
            }
        }
    }
    fclose(f);
    
    f = fopen("bias_layer8_new.txt", "w");
    if (f) {
        fprintf(f, "%.10f", bias);
        fclose(f);
    }
}

int main() {
    load_data();
    init_weights();
    
    printf("Training for %d iterations...\n", NUM_ITERATIONS);
    
    double loss_sum = 0;
    int report_interval = 100000;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Pick random sample
        int f_idx = rand() % FRAMES;
        int u = rand() % TARGET_ROWS;
        int v = rand() % TARGET_COLS;
        
        // Target
        double target = targets[f_idx * TARGET_ROWS * TARGET_COLS + u * TARGET_COLS + v];
        
        // Forward Pass with Dropout
        double pred = bias;
        
        // Identify valid input range and kernel offsets
        // Output (u,v) = Sum Input[i,j] * Kernel[u-2i, v-2j]
        // u - 2i = kr => 2i = u - kr.
        // kr in [0, 8]. implies u - 8 <= 2i <= u.
        // Also 0 <= i < ROWS.
        
        // We will store contributing indices to update weights later
        // Structure: index into input, index into kernel, channel
        // But allocating arrays per pixel is slow?
        // We can just re-compute for backward pass? Or update online?
        
        // To handle dropout efficiently: iterate channels. Decide if keep or drop.
        // If keep, accumulate prediction and update weights.
        // But we need the TOTAL error to update weights.
        // So Frame: 
        // 1. Calculate Pred (only using kept channels).
        // 2. Calculate Error.
        // 3. Update Kept Channels Weights.
        
        // Precompute coordinate ranges
        int start_kr = (u % 2 == 0) ? 0 : 1;
        int start_kc = (v % 2 == 0) ? 0 : 1;
        
        // Just store the sum
        double channel_contribs[CHANNELS]; // Optimization: only compute if kept
        int kept[CHANNELS];
        
        for (int c = 0; c < CHANNELS; c++) {
            if (rand_double() < DROPOUT_RATE) {
                kept[c] = 0;
                continue;
            }
            kept[c] = 1;
            
            double sum_c = 0;
            for (int kr = start_kr; kr < KERNEL_SIZE; kr += 2) {
                int input_row = (u - kr) / 2;
                if (input_row < 0 || input_row >= ROWS) continue;
                
                for (int kc = start_kc; kc < KERNEL_SIZE; kc += 2) {
                    int input_col = (v - kc) / 2;
                    if (input_col < 0 || input_col >= COLS) continue;
                    
                    double val = inputs[(long)f_idx * CHANNELS * ROWS * COLS + 
                                        c * ROWS * COLS + 
                                        input_row * COLS + input_col];
                    sum_c += val * weights[c][kr][kc];
                }
            }
            pred += sum_c;
        }
        
        double error = pred - target; // Derivative of 0.5*(pred-target)^2 is (pred-target)
        loss_sum += error * error;
        
        // Update Weights and Bias
        bias -= LR * error;
        
        for (int c = 0; c < CHANNELS; c++) {
            if (!kept[c]) continue;
            
            // Re-iterate valid kernel positions to update
            for (int kr = start_kr; kr < KERNEL_SIZE; kr += 2) {
                int input_row = (u - kr) / 2;
                if (input_row < 0 || input_row >= ROWS) continue;
                
                for (int kc = start_kc; kc < KERNEL_SIZE; kc += 2) {
                    int input_col = (v - kc) / 2;
                    if (input_col < 0 || input_col >= COLS) continue;
                    
                    double val = inputs[(long)f_idx * CHANNELS * ROWS * COLS + 
                                        c * ROWS * COLS + 
                                        input_row * COLS + input_col];
                    
                    weights[c][kr][kc] -= LR * error * val;
                }
            }
        }
        
        if ((iter + 1) % report_interval == 0) {
            printf("Iter %d, MSE: %.6f\n", iter + 1, loss_sum / report_interval);
            loss_sum = 0;
        }
    }
    
    save_weights();
    printf("Training done.\n");
    return 0;
}
