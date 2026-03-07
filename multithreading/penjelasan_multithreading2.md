# Penjelasan Detail Alur Kerja `multithreading2.c`

Program `multithreading2.c` adalah implementasi peningkat resolusi video (*Super-Resolution*) untuk format YUV menggunakan model neural network **FSRCNN** secara spesifik dan diakselerasi performanya menggunakan pendekatan **Pipeline Multithreading** dengan pustaka `pthread`.

Secara garis besar, alih-alih satu thread mengerjakan satu frame gambar dari Layer 1 sampai Layer 8, program ini membuat layaknya *"Pabrik Assembly Line"*. Setiap **Layer AI** punya pekerja (Thread) masing-masing, dan gambar akan dioper dari satu pekerja ke pekerja lainnya menggunakan sistem antrean (Queue).

Berikut adalah penjelasan rinci dari mulai `main()`, kemana alirannya berjalan, serta fungsi-fungsi utamanya:

---

## 1. Alur Eksekusi Utama (`main()`)
Fungsi `main()` bertindak sebagai **Manager** (dan *Producer*) yang mempersiapkan lingkungan, memuat data AI, mengatur pekerja, lalu mulai menyuapi data video ke dalam pabrik.

**Step-step di dalam fungsi `main()`:**
1. **Validasi Argumen & Pembukaan File:** 
   Memeriksa apakah program dijalankan dengan parameter `input.yuv` dan `output.yuv`. Membuka file input untuk dibaca dan membuka file output untuk ditulis.
2. **Memuat Bobot (Weights) & Bias AI:**
   Program membaca file eksternal `.txt` (contoh: `weights_layer1.txt`, `biasess_layer1.txt`) yang berisi angka-angka pintar (bobot hasil training neural network). Ini dimuat ke dalam array global pada memori (RAM).
3. **Pre-Baca Komponen Chroma (U dan V):** 
   Karena algoritma FSRCNN disini **hanya memproses komponen warna Luma/Kecerahan (Y)**, maka komponen Chroma (U dan V) dari seluruh bagian video langsung dibaca dan disimpan sementara (di-*backup*) di memori (`uv_store`). Nantinya di ujung alur, komponen ini dikeluarkan lagi untuk digabungkan.
4. **Pembuatan Antrean (Queue):** 
   Program menginisiasi 9 Antrean *Thread-safe* (Aman untuk Multithreading). Antrean ini bertindak sebagai jembatan *ban berjalan* di dalam pabrik:
   `q_in` -> `q12` -> `q23` -> `q34` -> `q45` -> `q56` -> `q67` -> `q78` -> `q_out`
5. **Menjalankan Pekerja Mode Pabrik (Stage Threads & Consumer Thread):**
   - Membuat **8 Thread Pekerja** via `pthread_create`, di mana Thread ke-1 bertanggungjawab untuk menghitung Layer 1 AI, Thread ke-2 untuk Layer 2, dan seterusnya. Masing-masing diberikan antrean keluar-masuk yang selaras.
   - Membuat **1 Thread Consumer**, yang letaknya berada di ujung terakhir antrean (`q_out`).
6. **Menyuapi Data ke Pabrik (Pipeline Processing):** 
   Thread `main` secara terus-menerus mengambil komponen `Y` dari 1 buah _frame video_, mengubah skalanya dari `[0-255]` menjadi `[0.0 - 1.0]`, memasukkannya ke struktur data `FrameBuffer`, lalu melemparkannya ke gerbang depan (`q_in`).
7. **Mengirim Sinyal Selesai (Sentinel Element):** 
   Setelah 150 frames selesai dimasukkan, `main` memasukkan satu data bayangan dengan `frame_id = -1`. Ini adalah pesan yang menyatakan: *"Kerjaan sudah habis, kalau kalian terima ini, teruskan pesannya dan lalu pulang/berhenti!"*.
8. **Tunggu Pekerjaan Selesai & Bersih-Bersih (Join & Cleanup):** 
   `pthread_join` menahan program agar tidak berhenti duluan sebelum semua Thread Pekerja dan Consumer mengakhiri tugasnya. Setelah semua kelar, memori sisa dibereskan (memanggil `free()` dan `destroy_queue()`).

---

## 2. Pekerja Pabrik: Fungsi `stage_thread`
Ini adalah fungsi yang dijalankan secara konstan pada 8 thread terpisah (Layer 1 sampai 8).

**Step-step di `stage_thread`:**
1. Mengambil (`dequeue`) satu Frame/Gambar dari `input_queue`.
2. Jika frame tersebut adalah Sinyal Berhenti (`frame_id == -1`), frame tersebut dilempar ke `output_queue` dan thread langsung mati (Break).
3. Melakukan inisialisasi ukuran / alokasi memori kosong untuk hasil perhitungannya nanti (Tergantung aturan ukuran pada masing-masing layer / `scale`).
4. **Eksekusi AI:** Thread ini memanggil fungsi layer yang sesuai tanggungjawabnya (misal `layer3()`). Pada layer ini terjadi perulangan *Matrix Multiplication* (Konvolusi spasial) dengan `imfilter` / `deconv` lalu distabilkan via aktivasi `PReLU`.
5. Frame input yang sudah tidak dibutuhkan langsung dibuang dari memori `release_buffer()`.
6. Memasukkan hasil pemrosesan (frame output yang posisinya sekarang berada pada tahap progres ke-*n*) menuju antrean berikutnya (`output_queue`).
7. Kembali ke step 1 (Mengulangi untuk image selanjutnya dari antrean).

---

## 3. Pekerja Terakhir: Fungsi `consumer_thread`
Berjalan sendirian, tugas fungsi ini mirip dengan "Tukang Packing dan Kurir". Posisinya berada pas di ujung pipa dari output antrean `Layer 8` (`q_out`).

**Step-step di `consumer_thread`:**
1. Mengambil (`dequeue`) hasil absolut gambar Y *(Luma)* dengan resolusi tinggi (dari `q_out`).  
   *(Berhenti secara otomatis saat membaca `frame_id == -1`)*
2. Mengembalikan nilai warna Luma dari format desimal ke bilangan baku *uint8* (dari rentang `0.0 - 1.0` dikali batas nilai maksimal standar kemudian dibulatkan menjadi interval `0 - 255`).
3. Menulis Data Luma (Y resolusi tinggi) tersebut ke File Video di *Storage Disk*.
4. **Restorasi Chroma:** Masih ingat memori `uv_store` yang diletakkan di RAM saat awal-awal? Disini thread mengambil data *U* dan *V* yang bersangkutan dengan ID dari frame ini.
5. Melakukan **Pembesaran Polos (Nearest Neighbor / Replikasi)** untuk komponen U dan V karena kedua tipe piksel ini tidak di-*upscale* oleh AI. Algoritmanya menduplikasi 1 baris pixel menjadi kelipatan ukurannya, baik lurus horizontal mau pun vertikal. Menulis data ini ke file video tersebut.
6. Membersihkan frame dari RAM. (Sehingga memori tidak menyentuh limit maksimal dan membocorkan penggunaan RAM).

---

## 4. Mekanisme Jembatan (Sinkronisasi Antrean / `Queue`)
Karena beberapa thread membaca dan meletakkan memori yang lokasinya sama di saat yang bersamaan (Concurrency), bisa beresiko terjadi kesalahan hitung/crash sistem (Dikenal sebagai *Race Condition*).

Untuk mensinkronkannya, pada struct `Queue` diberikan:
- **`pthread_mutex_t`**: Seperti kunci gembok gilir. Jika Thread 3 ingin masukkan frame ke Antrean 3-4, pintu Antrean 3-4 akan dikunci dulu. Jika di saat bersamaan Thread 4 mencoba mengambil frame dari pintu antrean yang sama, Thread 4 diminta mengantre sebentar sampai Thread 3 merilis gemboknya.
- **`pthread_cond_t`**: Digunakan saat Antrean terdeteksi sepenuhnya penuh (*Not Full*) ataupun terdeteksi sepenuhnya kosong (*Not Empty*). 
  - Jika kosong, Thread ditiarapkan untuk menghemat konsumsi operasi prosesor *(Wait)* sampai dibangunkan oleh sinyal (*Signal*) oleh Thread sebelumnya yang memastikan dia baru saja melempar barang kesana.
  - Begitu sebaliknya, jika antrean penuh (batas 8 slot), Thread penyuplai disuruh tidur dulu sampai antrean ini diambil barang pertamanya oleh Thread perantara didepannya (sehingga ada sisa slot 1/diklaim).

## 5. Ringkasan Kegunaan Algoritma per Layer AI (Fungsi Matematika)
- **`layer1()` s/d `layer7()`**: Algoritma ekstraksi fitur spasial video / FSRCNN Feature Extraction dan Non-Linear Mapping, melakukan perhitungan konvolusi matrix (`imfilter()`), menjumlahkan *bias*, kemudian dimodulasi nilainya oleh aktivasi Parametric ReLU (`PReLU()`).
- **`layer8()`**: Lapisan Deconvolution (`deconv()`). Ini adalah mekanisme *upscaling* yang sebenarnya, di mana vektor piksel yang telah dipahami oleh AI diekspansi secara ruang (spatial resolution expansion) menjadikannya gambar berukuran besar. Layer ini tidak menggunakan *PReLU* lagi.
- **`imfilter()` & `pad_image()`**: Algoritma pendukung yang melengkapi tepian layar dengan barisan tambahan nol/kopian (*padding*) agar matriks tidak *error* membaca batas *(out of bound)* saat *filter matrix* bergerak menggeser (*sliding window*) gambar tersebut.
