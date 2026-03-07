#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "syncpilot.h"

// =========================================================
// CONTOH PENGGUNAAN FRAMEWORK SYNCPILOT (Mock Video Encoder)
// =========================================================
// Pipeline ini meniru aplikasi render video yang terdiri dari:
// Tahap 0: Parsing & Decode Blur (Cepat)
// Tahap 1: Analisa Motion Vector (Sedang)
// Tahap 2: Kompresi H.265 / Render Berat (Sangat Lambat -> Bottleneck)
// Output : Consumer akan mengurutkan ulang hasil render berdasarkan frame.
// =========================================================

// Struktur data kustom buatan developer
typedef struct {
    char data_buffer[256];
    int complexity_score;
    double render_quality;
} MyGraphicFrame;

// ========== TAHAP 0 (DECODE) ==========
void stage_0_decode(PipelineTask *task) {
    // Ambil data kita dengan aman dari framework (casting void* -> kustom)
    MyGraphicFrame *frame = (MyGraphicFrame*)task->data;

    sprintf(frame->data_buffer, "[DECODED]");
    // Simulasi mikrosekon pemrosesan cepat (10ms)
    usleep(10000); 
}

// ========== TAHAP 1 (MOTION ANALYSIS) ==========
void stage_1_motion(PipelineTask *task) {
    MyGraphicFrame *frame = (MyGraphicFrame*)task->data;

    char temp[256];
    sprintf(temp, "%s -> [MOTION_ANALYZED]", frame->data_buffer);
    strcpy(frame->data_buffer, temp);
    
    // Simulasi pemrosesan sedang (30ms)
    usleep(30000); 
}

// ========== TAHAP 2 (HEAVY ENCODE) ==========
void stage_2_encode(PipelineTask *task) {
    MyGraphicFrame *frame = (MyGraphicFrame*)task->data;

    char temp[256];
    sprintf(temp, "%s -> [ENCODED_H265]", frame->data_buffer);
    strcpy(frame->data_buffer, temp);
    
    // Simulasi pemrosesan SANGAT LAMBAT (Bottleneck: 150ms)
    // Walaupun lambat, framework akan memerintahkan semua Worker 
    // untuk mengeroyok tahap ini sehingga tidak terjadi pipeline stall.
    usleep(150000); 
}

// ========== KONSUMEN AKHIR (PENGURUT) ==========
void final_writer(PipelineTask *task) {
    MyGraphicFrame *frame = (MyGraphicFrame*)task->data;

    // Fungsi ini dijamin 100% dipanggil berurutan oleh Reorder Buffer 
    // dari ID 0, 1, 2, ... meskipun tahap Render H265 selesainya acak.
    printf("MENYIMPAN KE DISK -> Frame %02d | Isi: %s\n", task->task_id, frame->data_buffer);

    // Bebaskan memori payload yg kita ciptakan (wajib mencegah leak)
    free(frame);
}


// ================== PROGRAM UTAMA ==================
int main() {
    printf("=== MEMULAI TEST FRAMEWORK SYNCPILOT ===\n\n");

    int total_frames = 20;

    // 1. Definisikan Konfigurasi Pipeline
    PipelineConfig cfg;
    cfg.num_workers              = 4;   // Kita pakai 4 Thread Pekerja
    cfg.num_stages               = 3;   // Decode, Motion, Encode
    cfg.total_tasks              = total_frames; 
    cfg.queue_capacity_per_stage = 10;  

    // Hubungkan fungsi kustom tahap kita ke Framework
    cfg.stages[0] = stage_0_decode;
    cfg.stages[1] = stage_1_motion;
    cfg.stages[2] = stage_2_encode;
    
    // Hubungkan Consumer Penulis
    cfg.consumer  = final_writer;

    // 2. Start Engine Load Balancer Asinkron
    PipelineEngine *engine = pipeline_start(&cfg);
    if(!engine) {
        printf("Gagal memulai engine!\n");
        return 1;
    }

    printf("Pekerja %d siap. Mem-feeding %d Frame...\n\n", cfg.num_workers, total_frames);

    // 3. Masukkan Data Secara Berurutan (Thread Utama sbg Produser)
    for (int i = 0; i < total_frames; i++) {
        MyGraphicFrame *baru = (MyGraphicFrame*)malloc(sizeof(MyGraphicFrame));
        baru->complexity_score = i * 10; 
        
        // Lempar pekerjaan kita ke mulut tahapan 0 framework
        pipeline_feed(engine, i, baru);
    }

    // 4. Tutup pintu masuk (agar engine tahu kapan harus bunuh diri)
    pipeline_close_input(engine);

    // 5. Tunggu semuanya selesai dan bersihkan memori
    pipeline_wait_and_destroy(engine);

    printf("\n=== PROSES RENDER SELESAI ===\n");
    return 0;
}
