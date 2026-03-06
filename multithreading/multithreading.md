# Penjelasan Pipeline FSRCNN (`multithreading2.c`) & Persiapan Orange Pi 5

Prinsip kerja `multithreading2.c` berpusat pada **Pipeline Multithreading** dengan arsitektur Producer-Consumer. Berbeda dengan pendekatan sebelumnya (`fsrcnn_parallel.c`) yang mengeksekusi satu frame sekaligus dan me-multi-thread operasi matriksnya menggunakan OpenMP, `multithreading2.c` memutarbalikkan logikanya menjadi sebuah "Pabrik Perakitan".

Berikut adalah penjelasan cara kerjanya secara berurutan:

## 1. Arsitektur Pipeline (Thread per Layer)
Di `main`, kita membuat 8 buah *thread* (Pthread), di mana masing-masing thread **bertanggung jawab khusus untuk satu layer saja** selamanya.
- **Thread 1** hanya mengerjakan Layer 1.
- **Thread 2** hanya mengerjakan Layer 2.
- ... dan seterusnya sampai Thread 8.

Karena ada 8 thread yang berjalan bersamaan, maka di suatu waktu tertentu:
- Thread 8 mungkin sedang memproses **Frame 1**
- Thread 7 sedang memproses **Frame 2**
- ...
- Thread 1 sedang memproses **Frame 8**

Ini membuat CPU selalu sibuk. 8 Core bisa memproses 8 frame di tahap (layer) yang berbeda secara simultan.

## 2. Mekanisme Komunikasi Antar Layer (Thread-Safe Queue)
Antara setiap thread / layer, terdapat sebuah sistem "Antrian" (Queue).
`[Main/Input] → (q_in) → [Thread L1] → (q12) → [Thread L2] → (q23) → ... → [Thread L8] → (q_out) → [Consumer Thread / Output]`

Setiap queue dibatasi kapasitasnya (saat ini `capacity = 8`). Penggunaannya dilindungi oleh **Mutex** dan **Condition Variables** bawaan `pthreads`:
- Jika sebuah queue kosong, thread di depannya (yang butuh data) otomatis akan "tidur" (`pthread_cond_wait`) sampai dikabari ada data masuk. Ini menghemat penggunaan CPU.
- Jika sebuah queue penuh, thread di belakangnya (yang memproduksi data) juga akan "tidur" sampai antrian tersebut lapang.

## 3. Eksekusi Sekuensial di Dalam Layer (Tanpa OpenMP)
Kita telah menghapus `#pragma omp parallel for` di dalam layer 1 sampai 8 di file ini. Kenapa? Karena secara default OMP thread stack itu sangat kecil (< 256KB). Jika 8 thread *sedang berjalan* lalu tiap thread tiba-tiba melahirkan thread OMP secara internal (misalnya 4 thread tiap layer), total kita akan mendapatkan 8 × 4 = 32 thread berebut CPU dan memory space, yang berujung pada tingginya *context switching* dan berisiko *stack overflow* (`SIGBUS`).

Oleh karena itu, di `multithreading2.c`, setiap layer murni dieksekusi maju (sekuensial). Paralelismenya berasal dari *Concurrency Frame* di pipeline, bukan membelah operasi konvolusinya.

## 4. Consumer Thread
Hasil Layer 8 ditaruh di `q_out`. Sebuah `consumer_thread` khusus di ujung bertugas terus menerus menarik data (`dequeue`), mengalikan nilai desimal dengan 255.0, menjahit kembali kanal *Luminance* (Y) dengan *Chrominance* (U dan V) yang replikasinya tadi dicadangkan di memori (`uv_store`), dan akhirnya menulis ke file tujuan `.yuv`. Menaruh tugas *I/O (Input/Output write)* ke thread terpisah ini sangat penting agar Thread L8 tidak membuang waktu menunggu antrean / IO blocking dari Storage, sehingga mencegah *Deadlock* yang sebelumnya sempat terjadi.

---

# Persiapan Eksekusi / Porting ke Orange Pi 5 (RK3588)

Kamu memiliki target board Orange Pi 5 yang berbasis arsitektur ARM *big.LITTLE* (terdiri dari 4 Core Cortex-A76 yang kencang, dan 4 Core Cortex-A55 yang hemat daya/pelan). Di sisi lain (Mac/PC x86), semua core relatif homogen kinerjanya. amati poin-poin berikut saat beralih ke OPi 5:

### 1. CPU Affinity & Penempatan Thread (Sangat Krusial)
Karena prosesor asimetris (big.LITTLE), sangat mungkin Linux OS melempar Thread 8 (layer 8 yang paling berat, deconv 9x9) ke *Cortex-A55 (Core pelan)* yang menganggur, dan layer 2 yang ringan ke *Cortex-A76 (Core kencang)*. Hal ini akan menyebabkan *bottleneck* parah di L8 dan antrean pipeline macet.
- **Tindakan:** Aktifkan kembali fungsi *POSIX CPU Affinity* (melalui modul Linux `<sched.h>` dengan `sched_setaffinity` dan macro `CPU_SET`) pada awal inisiasi setiap *Thread Layer*.
- **Contoh:** Modifikasi fungsi `stage_thread` agar Thread untuk Layer 8, 3, 4, 5, 6 (layer berat) dipaksa terikat/pinning pada processor ID milik kluster Cortex-A76 (umumnya CPU ke 4 s/d 7). 

### 2. Kapasitas Queue & Batasan Memori ARM Board
Meskipun OPi 5 kuat, manajemen beban IO dan RAM LPDDR4x-nya tidak setangguh workstation macOS.
- **Tindakan:** Amati kestabilan sistem dari kapasitas `create_queue(8)`. Jika program dibunuh secara sepihak oleh kernel Linux karena kehabisan RAM (`OOM-Killer`), kurangi ukuran limit antrean menjadi `4` atau `2`. Tujuannya untuk mengerem pipeline input agar tidak terlalu rakus menyimpan *FrameBuffer* gantung di RAM saat back-pressure tinggi.

### 3. Memunculkan Kembali Karakteristik *Race Condition* Sejati
Tujuan akhirmu mendeploy ini adalah melatih/memonitor model (*Micro-Compensator*) atas kegagalan (anomali) hardware spesifik. 
Saat ini *queue* / sistem antrean sangat steril dan sangat sinkron dilindungi mutlak oleh Mutex. Dan `imadd` pada array hasil juga disengaja terjadi non-deterministik namun masih "rapi".
- **Tindakan:** Jika model-mu tidak menangkap cukup anomali yang diharapkan di OPi 5, longgarkan perlindungan konvolusi deconv atau modifikasi alur `enqueue`/`dequeue` (hilangkan satu gembok pthread lock) sebagai "injeksi noise / balapan data murni" yang memang sengaja diciptakan untuk data pelatihannya (*generate_race_dataset*).
