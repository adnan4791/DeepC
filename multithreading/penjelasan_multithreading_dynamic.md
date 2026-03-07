# Penjelasan Detail Fungsi `main()` pada `multithreading_dynamic.c`

Berbeda dengan model pipelining sekuensial pada versi sebelumnya, `multithreading_dynamic.c` menggunakan arsitektur **Worker Pool (Pasukan Pekerja)** dengan antrean tugas terpusat menggunakan prioritas (Priority Queue) dan **Reorder Buffer**.

Berikut adalah alur logika yang terjadi di dalam fungsi `main()` secara berurutan:

## 1. Persiapan Konfigurasi Dasar dan Validasi Argumen
Sistem memvalidasi argumen yang masuk (harus ada letak file `.yuv` untuk *input* dan letak penyimpanan *output*). Parameter lain masih bersifat *hardcoded* untuk keperluan penelitian: 
- `scale = 2` (diperbesar 2x lipat)
- `inCols = 176`, `inRows = 144` (format QCIF)
- `numFrames = 150` (menguji proses pada 150 *frame* di video Suzie)

Selanjutnya fungsi memuat atau menyalin (*load*) file `.txt` yang berisi himpunan **Bobot (Weights)** dan **Bias** untuk model FSRCNN ke dalam array global pada RAM secara berurutan (Layer 1 sampai 8).

## 2. Prafetch (Pre-load) Komponen Chroma (U dan V) ke dalam RAM
Karena model kecerdasan buatan (FSRCNN) pada penelitian ini **hanya menyempurnakan resolusi citra hitam putih/terang** (Kanal Luma / Y), sedangkan warna asli (Kanal Chroma / UV) tidak perlu masuk AI dan hanya diedit menggunakan *interpolation bi-linear* ganda (2x *upscale*):
- Kode membaca habis seluruh isi video terlebih dulu hanya untuk mencuri porsi data U dan V dan memasukkannya sementara ke laci RAM di struktur `uv_store` sebanyak jumlah 150 *frame*.
- Setelah semua warna U dan V dicuri, kursor file (`inFp`) dikembalikan ke baris awal banget `rewind(inFp)` agar sistem nantinya bisa pelan-pelan membaca data citra Y (Luma) per *frame*.

## 3. Menciptakan Gawang Tugas (Work Pool) Global
Jika sebelumnya masing-masing perantara Layer punya *Queue* / antrean khusus (seperti pipa terhubung titik demi titik), sekarang semuanya disatukan ke dalam satu "Papan Pengumuman" global bernama `WorkPool`:
- `pool.layer_q[0]` berarti tempat membuang *frame* yang siap dikerjakan oleh **Layer 1**.
- `pool.layer_q[7]` berarti tempat membuang *frame* yang selesai Layer 7 dan bersiap dikerjakan *Worker* mana pun untuk **Layer 8**.
- Semua *thread* berbagi **satu anak kunci yang sama** (`pool.lock`) tiap kali ingin melirik pekerjaan di papan pengumuman. Terdapat sinyal ganda (`cond_work` dan `cond_space`) yang akan menjerit/memanggil *(signal)* pekerja yang sedang tertidur jika kebetulan ada tugas baru menumpuk.

## 4. Membangun Reorder Buffer (Alat Pengurut Salinan Output)
Karena pekerja bekerja saling tunjang dan berlomba, *Frame* yang duluan selesai ditebak/divalidasi oleh FSRCNN Layer 8 mungkin saja hasil dari `Frame 10` sementara `Frame 9` malah belakangan. Oleh karena itu, *main()* menyiapkan:
`reorder.slots = (FrameBuffer**)calloc(numFrames, sizeof(FrameBuffer*));`
Laci loker ini dipakai untuk menyimpan sementara *frame* hasil dari pekerja (Worker) sampai saatnya tepat bagi kurir (*Consumer*) untuk mengemas dan mengurutkannya (baca di poin ke-5). 

## 5. Melahirkan Consumer Thread (Si Konsumen/Penulis Akhir Berurutan)
Thread *Consumer* (1 thread tunggal) dilahirkan pertama kali sejenak sebelum *Worker* bekerja.
Ia hanya diam dan memantau loker `Reorder Buffer`:
> "Apakah dokumen `Frame 1` sudah selesaikan dibuatkan suratnya di slot [1]?"
Jika slot [1] sudah berisi data *(tidak lagi NULL)*, ia akan memungutnya, mengalikan 255 (format Float menjadi skala *Color Depth* Piksel 8-bit), menggabungkannya dengan warna yang disimpan di laci `uv_store` poin #2, lalu mencetaknya ke *hard disk*. Lalu dia akan menunggu dokumen `Frame 2` secara urut, tidak memedulikan jika dokumen `Frame 10` ada di loker!

## 6. Melahirkan Pasukan Pekerja (Worker Threads)
Kode melakukan *looping* untuk memanggil fungsi `pthread_create` membuat `NUM_WORKERS` (biasanya disetel ke 8 secara merata), yang semuanya mengikat fungsi `worker_thread`. Mereka langsung terjun ke dalam struktur balapan algoritma *Dynamic Load Balancing* yang berputar selamanya sampai tak ada *frame*.

## 7. Membaca Video dan Mengumpankan ke Layer 1
Sistem me-*looping* dari 0 sampai 149 (150 Frame). Thread utama di dalam block (*Main*) membaca data saluran Y dari file input satu-persatu per *frame*:
- Data diubah dari derajat abu-abu bernilai `0 - 255` menjadi koma (`0.0 - 1.0`).
- *Frame* tersebut dibungkus menjadi bungkusan surat struct `FrameBuffer`.
- Disisipkan (*push*) melalui fungsi gembok `sq_push(&pool.layer_q[0], fb)`.
- Karena ia dilempar ke `layer_q[0]`, pekerjaan ini dilabeli sebagai **Layer 1**.
- Seketika di dalam *WorkPool*, *Main thread* menekan bel / sinyal (`pthread_cond_broadcast`) membangunkan semua *Worker* yang diam tak peduli dengan isyarat: *"Hei bangun, ada setoran tugas baru!"*. Setoran ini dikerjakan di balik layar sambil fungsi utama *main* membaca frame selanjutnya terus-menerus.

Setelah seluruh 150 kotak *frame* habis masuk ke laci `layer_q[0]`, maka kunci khusus `pool.input_done = 1;` ditandai untuk isyarat berhenti (*sentinel* sinyal mati).

## 8. Pembersihan (Cleanup & Sinkronisasi)
`pthread_join` menahan *Main* agar program C utama tidak buru-buru tamat. Sistem menunggu seluruh pasukan *Worker* lelah dan menyadar tidak ada kerjaan (kembali dengan fungsi `NULL`), kemudian menunggu kurir / ` consumer thread` selesai mengekskresi seluruh *frame* akhir ke file `output.yuv`. 

Akhirnya memori yang sebelumnya dipinjam dikembalikan (*free/destroy*), mutex dihancurkan untuk membebaskan RAM inti / *kernel space*. Kemudian mencetak `Selesai.` ke *log* layar sebagai titik terminasi (*exit 0*).
