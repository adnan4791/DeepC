// fsrcnn_race_sim.c
#include <omp.h>
#include <stdlib.h>

// Fungsi imadd ASLI Anda (tanpa perlindungan critical section)
void imadd(double *img_fltr_sum, double *img_fltr_crnt, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            int cnt = i * cols + j;
            // RACE CONDITION TERJADI DI BARIS INI KARENA DIPANGGIL BERSAMAAN OLEH MULTI-THREAD
            img_fltr_sum[cnt] = img_fltr_sum[cnt] + img_fltr_crnt[cnt]; 
        }
}

// Fungsi ekspor yang akan dipanggil oleh PyTorch
void simulate_layer8_race(double *input_56_channels, double *output_1_channel, int out_rows, int out_cols, int num_channels)
{
    int frame_size = out_rows * out_cols;
    
    // Inisialisasi output dengan 0
    for (int i = 0; i < frame_size; i++) {
        output_1_channel[i] = 0.0;
    }

    // Loop OpenMP persis seperti di source code asli FSRCNN Anda
    #pragma omp parallel for
    for (int j = 0; j < num_channels; j++)
    {
        // Ambil pointer ke channel ke-j (hasil deconv dari PyTorch)
        double *current_channel = input_56_channels + (j * frame_size);
        
        // Panggil imadd yang akan menyebabkan race condition di variabel output_1_channel
        imadd(output_1_channel, current_channel, out_cols, out_rows);
    }
}