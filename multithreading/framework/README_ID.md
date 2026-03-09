# 🚀 SyncPilot Framework
**Generic Dynamic Load Balancing & Pipeline Reordering Framework for C/C++**

**SyncPilot** adalah kerangka kerja (*framework*) C ringan yang didesain secara khusus untuk mengeksekusi proses komputasi berat bertahap (pipelining) menggunakan konsep **Asynchronous Worker Pool**, **Priority Queues**, dan penjaminan luaran berurutan menggunakan **Reorder Buffer**.

Framework ini awalnya diarsiteki untuk menyokong performa algoritma Deep Learning FSRCNN (Fast Super-Resolution Convolutional Neural Network) pada Arsitektur ARM (big.LITTLE SoC), namun kini diabstraksikan sehingga 100% *generik* dan siap digunakan untuk skenario pipelining lainnya, seperti:
1. Video / Audio Encoding (FFmpeg-like operations)
2. Rendering Grafis
3. Big Data Processing pipeline
4. Packet Inspection

---

## 🔥 Mengapa Menggunakan SyncPilot?
Dibandingkan membina *thread* masing-masing per tahapan (yang memicu *bottleneck* ketika ada tahapan yang sangat lambat), SyncPilot menelurkan pasukan **Homogenous Worker Threads**.

1. **Auto-Balancing & Zero Latency Profiling**: Jika `Tahap 3` butuh 500ms dan `Tahap 1` hanya 2ms, para Pekerja (Worker) tidak akan diam (idle) menunggu. Secara otomatis (lewat *Priority Queue*), pekerja akan langsung beramai-ramai menyerbu `Tahap 3` ketika mulai menumpuk.
2. **Berurutan tanpa Memblokir (Reorder Buffer)**: Akibat pekerja saling berlomba, hasil eksekusi pekerja menjadi tidak berurutan (*out-of-order execution*). Anda tidak butuh cemas, SyncPilot mencadangkan tempat khusus bernama `Reorder Buffer` yang membekukan dan secara rapi mengurutkan (*sort*) hasil tersebut `[0, 1, 2, 3...]` sebelum diserahkan pada sang Penulis Akhir (*Consumer Writer*).
3. **Mendukung Custom Struct (*void pointers*)**: Anda bebas melempar bentuk data Struct apapun ke mesin ini karena data dipassing secara internal menggunakan pointer *generik* `void *data`.

---

## ⚙️ Arsitektur Logika Sistem

![SyncPilot Architecture](../docs/hierarki.jpeg)

```text
[ Developer Feed() ] --> [ Tahap 0 ] --> [ Tahap 1 ] --> [ Tahap N (Akhir) ]
                               \              /                 |
                                \            /                  |
                             [ N Worker Threads ]               v
                                                        [ Reorder Loker ]
                                                                |
 [ Developer Writer() ] <--------------------------------[ Consumer Kurir ]
```

---

## 🛠️ Cara Penggunaan (Tutorial Singkat)

### 1. Masukkan *Header* dan Atur Struktur Data Anda
```c
#include "framework/syncpilot.h"

// Bebas membuat tipe data Struct apapun untuk disalurkan antar Tahap!
typedef struct {
    int frame_id;
    char text[256];
    double image_data[1024]; 
} MyPayload;
```

### 2. Definisikan Fungsi Pekerjaan Per Tahap (Stages)
Setiap tahapan (*stage*) yang dipanggil akan menerima data `PipelineTask *task`. Ekstrak struktur Anda dari parameter `task->data`.

```c
void stage_0_read(PipelineTask *task) {
    MyPayload *payload = (MyPayload*)task->data;
    sprintf(payload->text, "Tahap 0 Selesai");
    // Lakukan komputasi...
}

void stage_1_encode(PipelineTask *task) {
    MyPayload *payload = (MyPayload*)task->data;
    sprintf(payload->text, "Tahap 1 Encode Selesai");
    // Lakukan komputasi mahaberat...
}
```

### 3. Definisikan Konsumen Penulis (Writer)
Fungsi ini spesial! Fungsi ini dijamin oleh *framework* pasti akan dipanggil secara **berurutan (Chronological)**, dari `Task ID 0`, lalu `1, 2, 3..`. 
Di sinilah Anda menulis output ke `FILE` / Terminal dengan aman tanpa data tumpang-tindih. Begitu selesai, **Anda wajib membersihkan memori** `payload`.

```c
void final_writer(PipelineTask *task) {
    MyPayload *payload = (MyPayload*)task->data;
    printf("Simpan Ke Disk Tuntas! ID: %d | Isi: %s\n", task->task_id, payload->text);
    
    free(payload); // BEBASKAN MEMORI KASTEM ANDA!
}
```

### 4. Nyalakan Engine (Main Function)
```c
int main() {
    PipelineConfig cfg;
    cfg.num_workers              = 8;   // Jumlah thread pekerja bersamaan
    cfg.num_stages               = 2;   // Kita hanya punya read & encode
    cfg.total_tasks              = 100; // Jumlah total item yg mau diproses
    cfg.queue_capacity_per_stage = 16;  // Buffer antrean stage

    // Daftarkan fungsi Anda
    cfg.stages[0] = stage_0_read;
    cfg.stages[1] = stage_1_encode;
    cfg.consumer  = final_writer;

    // Hidupkan mesin!
    PipelineEngine *engine = pipeline_start(&cfg);

    // Lempar pekerjaan masuk!
    for (int i = 0; i < 100; i++) {
        MyPayload *baru = (MyPayload*)malloc(sizeof(MyPayload));
        baru->frame_id = i;
        
        pipeline_feed(engine, i, baru);
    }

    // Isyaratkan pintu masuk ditutup (Tidak ada data lg)
    pipeline_close_input(engine);

    // Join dan tunggu sampai selesai beserta pembersihan API internal
    pipeline_wait_and_destroy(engine);
    
    return 0;
}
```

## 📜 Kompilasi (Compile)

Karena *framework* ini menggunakan arsitektur bawaan sinkronisasi multi-tugas POSIX Threads, Anda WAJIB mewariskan bendera (*flag*) pustaka pthread atau OpenMP.

```bash
# Contoh dengan file main Anda
gcc -O3 -o my_app main.c framework/syncpilot.c -lpthread -Wall
```

Selesai! Anda hanya perlu fokus memikirkan fungsionailtas matematika komputasi di kode C Anda tanpa memikirkan rumitnya *deadlock*, *mutex lock*, *condition signal broadcast*, atau manajemen *threadpool* lagi!
