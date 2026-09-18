# Kontribusi & Studi Literatur — Optimalisasi Granularitas Penjadwalan Paralel via Loop/Register Blocking (Tesis 2)

> Dokumen ini **terpisah** dari
> [`KONTRIBUSI-DAN-REFERENSI.md`](./KONTRIBUSI-DAN-REFERENSI.md) (Tesis 1, soal *layout memori*
> CHW vs HWC). Tesis 1 menjawab **DI MANA** data feature-map disimpan di memori. Dokumen ini
> menjawab pertanyaan yang berbeda: **BAGAIMANA loop konvolusi dipecah menjadi satuan kerja
> paralel**, dan berlaku independen dari layout memori yang dipakai (bisa diterapkan di atas
> CHW maupun HWC).

## 0. Bahasa Manusia: Apa yang Sebenarnya Diubah di Kode

Sebelum revisi ini, satu "tugas" yang dibagikan penjadwal OpenMP ke tiap thread = satu baris
piksel keluaran. Supaya sebuah thread bisa mulai bekerja, ada ongkos administratif setiap kali
ia mengambil tugas baru dari penjadwal (dispatch, sinkronisasi antar-thread). Kalau ongkos itu
dibandingkan dengan kerja nyata dalam satu baris — terutama untuk layer berkernel kecil (1×1)
— porsinya jadi relatif besar: banyak "minta tugas, kerja sedikit, minta lagi", bukan
"kerja banyak sekali minta".

Perubahan pada `conv_layer_hwc` di `source.c` mengubah ukuran satu tugas itu: sekarang satu
tugas = **dua baris keluaran sekaligus** (`oi` dan `oi+1`), dikerjakan berurutan oleh thread
yang sama dalam satu kali "ambil tugas". Konsekuensinya:

1. **Jumlah tugas yang harus dibagi penjadwal berkurang setengah** (dari `rows` menjadi
   `rows/2`) — ongkos dispatch/sinkronisasi per satuan kerja ikut berkurang, karena ongkos itu
   terjadi per-tugas, bukan per-baris-piksel di dalamnya.
2. **Ukuran (granularitas) tiap tugas naik 2×** — tiap thread mengerjakan lebih banyak sebelum
   kembali meminta tugas baru, memperbaiki rasio *kerja nyata* : *overhead manajemen tugas*.
3. **Bonus reuse register**: nilai bobot kernel (`w = w_oc[ic*filtersize + k_idx]`) tidak
   bergantung pada baris — hanya pada kanal dan posisi di dalam kernel. Sekali diambil dari
   memori, nilai itu langsung dipakai dua kali (`sum0 += ...`, `sum1 += ...`) sebelum pindah
   ke elemen bobot berikutnya, sehingga jumlah pembacaan array bobot untuk kerja yang sama
   berkurang setengahnya.
4. Baris ganjil terakhir (bila `rows` ganjil) ditangani lewat flag `two_rows` agar tidak
   menulis di luar batas array.

Poin penting: **ini sama sekali tidak menyentuh soal layout memori** (masih HWC, mengikuti
Tesis 1) — ini murni transformasi *jadwal loop* (loop schedule/tiling), yang bisa saja
diterapkan di atas layout CHW lama sekalipun. Itu sebabnya ini layak berdiri sebagai
pertanyaan riset kedua yang independen.

## 1. Landasan Masalah: Mengapa Ini Menarik Diteliti

Ada dua isu klasik komputasi paralel yang terpisah dari isu bandwidth memori (yang jadi fokus
Tesis 1):

**(a) Masalah granularitas (*grain-size problem*) pada penjadwalan loop paralel.** Setiap kali
runtime OpenMP membagikan sebuah chunk iterasi ke sebuah thread, ada biaya non-nol untuk itu
(pengecekan/pembaruan indeks bersama, potensi cache-line bouncing di titik pembagian tugas,
laten bangun-thread pada beberapa scheduler). Jika ukuran kerja per-chunk terlalu kecil
dibanding biaya ini, mempercepat program dengan menambah core mentok bukan karena bandwidth
memori atau kapasitas komputasi, tetapi karena **overhead manajemen paralelisme itu sendiri**
— isu klasik yang dibahas Polychronopoulos & Kuck (1987) dalam merancang skema *guided
self-scheduling* untuk menyeimbangkan beban kerja sekaligus meminimalkan overhead sinkronisasi.

**(b) Reuse operand lewat *loop/register blocking*, independen dari layout data.** Bahkan
tanpa mengubah tempat data disimpan, mengelompokkan (*blocking*/*tiling*) beberapa unit
komputasi agar memakai ulang nilai yang sama sekali-muat sebelum di-*evict* adalah teknik
optimisasi HPC klasik — dipakai pertama kali secara sistematis untuk cache blocking pada
algoritma matriks (Lam, Rothberg, & Wolf, 1991) dan untuk *register* blocking pada kernel GEMM
performa tinggi (Goto & van de Geijn, 2008; dasar dari GotoBLAS/OpenBLAS). Prinsipnya sama
persis dengan yang diterapkan di `conv_layer_hwc`: satu operand (bobot kernel) yang sudah
dimuat dipakai untuk lebih dari satu akumulator sebelum dibuang.

**(c) Skedul sebagai sumbu optimisasi yang terpisah dari algoritma.** Ragan-Kelley et al.
(2013), lewat bahasa dan compiler *Halide*, memformalkan bahwa untuk kernel bertipe *stencil*
(termasuk convolution — jendela geser di atas array multidimensi, persis pola FSRCNN), ada dua
hal yang bisa dioptimasi terpisah: **algoritma** (apa yang dihitung) dan **jadwal/schedule**
(tiling, unrolling, urutan loop, granularitas paralelisasi — kapan dan bagaimana itu dihitung).
Row-pair blocking pada `conv_layer_hwc` adalah contoh konkret optimisasi *schedule* murni:
algoritmanya (rumus convolution + PReLU) sama sekali tidak berubah — dan itulah yang
membuktikan (secara bit-exact) bahwa ini murni soal jadwal, bukan soal hasil.

## 2. Kontribusi Tesis

1. **Mengidentifikasi *grain-size problem* sebagai bottleneck yang berbeda dari bandwidth
   memori** — menunjukkan bahwa bahkan setelah layout data yang tepat dipilih (HWC, hasil
   Tesis 1), performa masih bisa ditingkatkan lebih jauh dengan mengatasi overhead
   penjadwalan OpenMP per-iterasi, sebuah sumbu optimisasi yang berbeda dan independen.
2. **Menerapkan row-pair (unroll-by-2 pada dimensi tinggi) sebagai transformasi *schedule*
   generik** pada kernel convolution FSRCNN — menghalve jumlah iterasi loop paralel dan
   melipatgandakan granularitas kerja per satuan tugas, tanpa mengubah algoritma komputasi.
3. **Menunjukkan reuse register (weight-reuse-via-blocking) sebagai sumbu optimisasi yang
   ortogonal terhadap layout memori** — teknik yang sama berlaku terlepas dari CHW atau HWC,
   dibuktikan dengan menumpuknya di atas baseline HWC dari Tesis 1 tanpa mengubah hasil
   (bit-exact) namun tetap memberi speedup tambahan yang independen dan terukur.
4. **Mengisolasi dan mengukur kontribusi murni loop blocking**, terpisah dari efek layout
   memori Tesis 1 (lihat tabel §3) — penting secara metodologis agar klaim performa tidak
   mencampur dua penyebab yang sebenarnya berbeda sumbunya.
5. **Membuka ruang desain untuk perluasan riset**: faktor blocking lebih besar (4/8 baris),
   *tiling* dua dimensi (baris×kolom sekaligus), trade-off tekanan register vs
   auto-vektorisasi compiler, dan potensi *auto-tuning* faktor blocking (mengikuti pendekatan
   pencarian jadwal ala Halide/TVM) untuk berbagai ukuran resolusi video dan jumlah core.
6. **Validasi numerik ketat (bit-exact)** terhadap versi tanpa blocking dan versi CHW asli,
   membuktikan transformasi ini murni soal jadwal eksekusi, bukan perubahan hasil komputasi.

## 3. Tabel Benchmark — Mengisolasi Kontribusi Loop Blocking

Data dari `suzie_qcif.yuv` (150 frame, 176×144→352×288, mesin uji 10-core). Kolom "HWC
piksel-paralel" adalah baseline Tesis 1 (tanpa blocking); kolom "+ row-pair blocking" adalah
hasil setelah menambahkan teknik pada dokumen ini di atas baseline yang sama.

| Threads | HWC piksel-paralel (baseline T1) | HWC + row-pair blocking | Speedup murni dari blocking | Speedup total vs CHW asli |
|---|---|---|---|---|
| 4  | 5,86 s | **4,51 s** | ~23% | ~29% |
| 8  | 5,35 s | **3,99 s** | ~25% | ~30% |
| 10 | 5,04 s | **3,93 s** | ~22% | ~33% |

Kolom "speedup murni dari blocking" adalah **kontribusi Tesis 2 yang terisolasi** —
konsisten di angka ~22–25% terlepas dari jumlah thread, menunjukkan efeknya lebih ke arah
pengurangan overhead per-iterasi & peningkatan reuse register (relatif stabil terhadap jumlah
core) dibanding efek Tesis 1 yang lebih bervariasi mengikuti jumlah thread (§2, Tesis 1).

## 4. Studi Literatur Terkait

| # | Referensi | Temuan Utama | Relevansi dengan Tesis Ini |
|---|---|---|---|
| 1 | Polychronopoulos, C. D., & Kuck, D. J. (1987). *Guided Self-Scheduling: A Practical Scheduling Scheme for Parallel Supercomputers*. IEEE Transactions on Computers, 36(12), 1425–1439. | Merumuskan trade-off *grain size* pada penjadwalan loop paralel: chunk terlalu kecil → overhead sinkronisasi dominan; terlalu besar → load imbalance. | Landasan teori langsung untuk klaim "mengurangi jumlah iterasi loop paralel = mengurangi overhead scheduling" pada §1(a) & kontribusi #1. |
| 2 | Lam, M. S., Rothberg, E. E., & Wolf, M. E. (1991). *The Cache Performance and Optimizations of Blocked Algorithms*. ASPLOS 1991, 63–74. | Blocking/tiling pada algoritma matriks meningkatkan reuse data pada hierarki memori sebelum data di-*evict*; ukuran blok sangat memengaruhi performa nyata. | Dasar konsep *blocking* sebagai teknik reuse yang independen dari representasi data (layout) — landasan konseptual row-pair blocking. |
| 3 | Goto, K., & van de Geijn, R. A. (2008). *Anatomy of High-Performance Matrix Multiplication*. ACM Transactions on Mathematical Software, 34(3), 12:1–12:25. [PDF](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf) | Register blocking pada GEMM (GotoBLAS): satu operand dimuat sekali, dipakai untuk banyak akumulator, mengamortisasi biaya pemuatan memori. | Landasan **paling langsung** untuk row-pair blocking di `conv_layer_hwc` — pola identik: satu bobot kernel dimuat sekali, dipakai untuk 2 akumulator (`sum0`, `sum1`). |
| 4 | Ragan-Kelley, J., Barnes, C., Adams, A., Paris, S., Durand, F., & Amarasinghe, S. (2013). *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines*. PLDI 2013. | Memisahkan "algoritma" dari "jadwal" (tiling, unrolling, urutan loop, granularitas paralel) sebagai dua sumbu optimisasi independen untuk kernel bertipe stencil/convolution. | Kerangka konseptual utama tesis ini: membenarkan bahwa loop blocking adalah kontribusi **schedule**, terpisah dari algoritma maupun layout data (Tesis 1). |
| 5 | Georganas, E., Avancha, S., Banerjee, K., Kalamkar, D., Henry, G., Pabst, H., & Heinecke, A. (2018). *Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures*. SC18. [arXiv:1808.05567](https://arxiv.org/pdf/1808.05567) | Kernel convolution CPU performa tinggi memakai *blocked, JIT-compiled* loop nests dengan faktor unroll/tiling yang disetel ke ukuran register/cache. | Menghubungkan teknik row-pair blocking (unroll faktor 2) ke praktik umum kernel convolution CPU produksi — juga dirujuk di Tesis 1 untuk sudut layout, di sini untuk sudut *tiling factor*. |
| 6 | Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., Cowan, M., Wang, L., Hu, Y., Ceze, L., Guestrin, C., & Krishnamurthy, A. (2018). *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*. OSDI 2018. | Ruang pencarian otomatis untuk parameter jadwal tensor (tiling, unrolling, vektorisasi, penempatan memori) pada berbagai target hardware. | Referensi untuk arah pengembangan lanjutan (kontribusi #5): faktor blocking optimal bisa dicari otomatis, bukan dipilih manual (2 baris) seperti pada implementasi saat ini. |

## 5. Rekomendasi Judul Tesis

Judul-judul ini sengaja dipisah dari tema layout memori (Tesis 1) dan berfokus pada
**granularitas penjadwalan paralel** dan **reuse register lewat loop blocking**.

| Opsi | Judul | Catatan |
|---|---|---|
| A (disarankan) | **"Optimalisasi Granularitas Penjadwalan Paralel dan Reuse Register melalui Teknik Loop Blocking pada Inferensi FSRCNN Berbasis OpenMP di CPU Multi-Core"** | Menyebut eksplisit dua konsep inti (granularitas penjadwalan + reuse register), teknik (loop blocking), platform (OpenMP/CPU multi-core), dan model (FSRCNN) — selaras langsung dengan §1–2. |
| B | "Analisis Trade-off Overhead Penjadwalan OpenMP dan Reuse Data melalui Teknik Row Blocking pada Percepatan Inferensi Convolutional Neural Network Ringan" | Menonjolkan sisi analisis/trade-off (grain-size vs reuse) sebagai fokus utama — cocok bila bab hasil ingin bertumpu pada studi sensitivitas faktor blocking (2, 4, 8 baris) dan jumlah thread. |
| C | "Penerapan Prinsip Register Blocking GEMM pada Kernel Convolution untuk Optimalisasi Inferensi CNN Ringan di CPU Multi-Core" | Menonjolkan garis keturunan teoretis dari literatur GEMM/BLAS (Goto & van de Geijn) sebagai kerangka utama — cocok jika pembimbing ingin tesis eksplisit memposisikan diri sebagai "adaptasi teknik BLAS klasik ke convolution". |
| D | "Desain Jadwal Loop (Tiling dan Unrolling) yang Independen dari Layout Data untuk Optimalisasi Inferensi CNN pada CPU Multi-Core" | Paling dekat dengan framing Halide (algoritma vs jadwal) — cocok jika ingin tesis dibingkai metodologis/generalizable ke model/layer lain, bukan hanya hasil optimisasi FSRCNN. |

**Rekomendasi: Opsi A.** Judul ini paling akurat menggambarkan apa yang sudah diimplementasikan
dan diverifikasi (bit-exact + speedup ~22–25% yang terisolasi dari efek layout), sekaligus
menegaskan bahwa penelitian ini adalah sumbu kontribusi **kedua** yang berdiri sendiri dari
Tesis 1 — bukan sekadar detail tambahan di dalamnya.
