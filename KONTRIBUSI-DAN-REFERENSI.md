# Kontribusi & Studi Literatur — Optimalisasi Pola Akses Memori pada Inferensi FSRCNN

> Catatan revisi: draf sebelumnya membingkai pekerjaan ini sebagai "konversi CHW ke HWC".
> Itu hanya *teknik implementasi*, bukan kontribusinya. Dokumen ini menulis ulang framing-nya:
> pertanyaan risetnya adalah **mengapa layer-layer tipis FSRCNN sangat terbatas oleh bandwidth
> memori saat diinferensi di CPU, dan bagaimana pola akses/reuse data memengaruhi hal itu** —
> HWC + paralelisasi spasial hanyalah satu instrumen untuk menguji dan memperbaikinya.

## 1. Landasan Masalah: Mengapa Ini Menarik Diteliti

Kinerja kernel komputasi pada CPU modern umumnya dibatasi oleh salah satu dari dua hal:
kapasitas komputasi (FLOP/s) atau bandwidth memori (byte/s) — ini yang diformalkan **Roofline
Model** (Williams, Waterman, & Patterson, 2009). Ukuran yang menentukan posisi sebuah kernel
pada model ini adalah **intensitas aritmetika (AI)** = FLOP dikerjakan / byte data yang
dipindahkan dari memori. Kernel dengan AI rendah tidak bisa dipercepat hanya dengan menambah
core — ia perlu mengurangi lalu-lintas memori atau meningkatkan *reuse* data (Sze et al., 2017).

FSRCNN (Dong et al., 2016) sengaja dirancang dengan lapisan *shrinking/mapping* yang sangat
sempit (12 kanal) diapit dua lapisan *pointwise* (1×1) yang mengubah lebar kanal (56↔12) — pola
yang secara struktural mirip *pointwise convolution* pada MobileNet (Howard et al., 2017), yang
dalam literatur efficient-CNN dikenal punya *reuse* spasial nol (kernel 1×1 tidak menggeser
jendela) sehingga rasio FLOP terhadap byte-nya rendah secara inheren.

**Perhitungan AI per layer (estimasi *compulsory traffic*, QCIF 176×144, `double` 8 byte)**
pada implementasi ini menunjukkan pola yang tidak trivial:

| Layer | Kernel | In→Out kanal | FLOP (≈) | Byte dipindah (≈) | AI (FLOP/byte) |
|---|---|---|---|---|---|
| 1 | 5×5 | 1→56 | 71,0 juta | 11,57 MB | **6,1** |
| 2 | 1×1 | 56→12 | 34,1 juta | 13,79 MB | **2,5** ← terendah |
| 3–6 | 3×3 | 12→12 | 65,7 juta /layer | 4,88 MB /layer | **13,5** |
| 7 | 1×1 | 12→56 | 34,1 juta | 13,79 MB | **2,5** ← terendah |

Temuan yang tidak jelas dari sekadar melihat jumlah parameter/FLOP: **kedua lapisan 1×1
(layer 2 dan 7), bukan lapisan 5×5 pertama atau empat lapisan 3×3 di tengah, adalah bagian
yang paling terikat bandwidth memori** pada jaringan ini. Ini konsisten dengan pengamatan umum
soal *pointwise convolution* di literatur efficient-CNN, tetapi belum pernah — sejauh
penelusuran literatur — dikuantifikasi secara spesifik untuk FSRCNN pada konteks inferensi CPU
untuk video real-time. Ini adalah **temuan/kebaruan yang layak jadi sumbu tesis**, bukan hanya
"mengganti layout array".

## 2. Kontribusi Tesis

1. **Karakterisasi arithmetic-intensity per-layer FSRCNN pada CPU** — analisis roofline
   empiris yang mengidentifikasi lapisan *pointwise* (1×1) sebagai titik paling terikat
   bandwidth memori dalam jaringan, bukan lapisan dengan kernel terbesar. Ini memberi dasar
   kuantitatif untuk memilih *di bagian mana* optimisasi pola akses memori paling berdampak,
   alih-alih mengoptimasi seluruh jaringan secara seragam.
2. **Menunjukkan bahwa pola akses (reuse) data, bukan hanya jumlah FLOP, menentukan performa
   nyata** — dengan mengubah urutan loop dan organisasi memori *feature map* (channel-interleaved,
   piksel-mayor) sehingga setiap byte yang dimuat dari memori dipakai ulang oleh seluruh kanal
   keluaran dalam satu iterasi, alih-alih dimuat ulang per kanal secara terpisah (skema lama:
   pad + filter + tambah per kanal, berulang untuk tiap kanal keluaran). Ini secara langsung
   menaikkan AI *tercapai* (bukan cuma AI teoretis) untuk lapisan-lapisan yang tadinya
   memory-bound, terutama layer 2 dan 7.
3. **Mengidentifikasi keterbatasan granularitas paralel yang spesifik-arsitektur FSRCNN** —
   karena skema paralelisasi lama membagi kerja per kanal keluaran, lapisan 2–6 (12 kanal)
   hanya dapat memanfaatkan maksimum 12 thread berapa pun jumlah core yang tersedia,
   independen dari isu layout memori di atas. Memaralelkan pada dimensi piksel (baris×kolom,
   puluhan ribu unit kerja) melepas keterikatan jumlah thread dari lebar kanal jaringan —
   relevan khusus untuk arsitektur ramping seperti FSRCNN, tidak umum dibahas di literatur
   NHWC/NCHW yang biasanya menguji jaringan lebar (ResNet, VGG).
4. **Validasi numerik ketat (bit-exact)** terhadap implementasi acuan pada beban kerja video
   nyata (YUV 4:2:0, 150 frame), membuktikan bahwa perbaikan yang diamati murni berasal dari
   pola akses memori/paralelisasi, bukan perubahan hasil komputasi — penting karena keluaran
   di sini adalah kualitas piksel video, bukan sekadar angka benchmark.
5. **Studi kasus pada implementasi C tanpa dependensi eksternal** (tanpa BLAS/oneDNN/TVM) —
   sebagian besar studi roofline/NHWC-NCHW dilakukan di atas backend inferensi yang sudah
   teroptimasi; menunjukkan efek yang sama pada kode yang ditulis manual relevan untuk
   deployment di perangkat *edge*/tertanam yang tidak punya pustaka DL.

> HWC + `#pragma omp parallel for collapse(2)` pada loop piksel (lihat `conv_layer_hwc` di
> `source.c`) adalah **instrumen** untuk kontribusi #2 dan #3 di atas, bukan kontribusi itu
> sendiri — penting untuk penulisan bab metode agar tidak terbaca sebagai "sekadar mengganti
> layout array".

## 3. Studi Literatur Terkait

### 3.1 Landasan Teoritis: Model Performa & Reuse Data

| # | Referensi | Temuan Utama | Relevansi dengan Tesis Ini |
|---|---|---|---|
| 1 | Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM, 52(4), 65–76. | Model roofline: performa kernel dibatasi oleh `min(FLOP/s puncak, AI × bandwidth memori)`. | **Kerangka teori utama** tesis ini — dasar untuk perhitungan AI per layer di §1 dan argumen "kenapa layout memori penting", bukan sekadar "HWC lebih cepat". |
| 2 | Sze, V., Chen, Y.-H., Yang, T.-J., & Emer, J. S. (2017). *Efficient Processing of Deep Neural Networks: A Tutorial and Survey*. Proceedings of the IEEE. [arXiv:1703.09039](https://arxiv.org/pdf/1703.09039) | Taksonomi *dataflow* CNN (mis. *row-stationary*) yang secara eksplisit memaksimalkan reuse data pada tiap level hierarki memori untuk mengurangi energi/lalu-lintas memori. | Landasan konseptual untuk mendesain `conv_layer_hwc` sebagai skema *dataflow* (bukan sekadar "layout"), dan kerangka untuk membahas trade-off reuse-vs-paralelisme di bab metode. |
| 3 | Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. [arXiv:1704.04861](https://arxiv.org/pdf/1704.04861) | Convolution *pointwise* (1×1) tidak punya reuse spasial (jendela kernel tidak bergeser), sehingga secara struktural punya AI rendah dibanding convolution berkernel besar. | Menjelaskan **mengapa** layer 2 & 7 FSRCNN (keduanya 1×1) muncul sebagai titik AI terendah pada tabel §1 — bukan kebetulan, tapi properti umum pointwise convolution. |

### 3.2 Dasar Arsitektur & Implementasi

| # | Referensi | Temuan Utama | Relevansi dengan Tesis Ini |
|---|---|---|---|
| 4 | Dong, C., Loy, C. C., & Tang, X. (2016). *Accelerating the Super-Resolution Convolutional Neural Network*. ECCV 2016. [arXiv:1608.00367](https://arxiv.org/pdf/1608.00367) | Paper asli FSRCNN: lapisan *shrinking-expanding* (56↔12 kanal) untuk mengurangi FLOP dan mempercepat inferensi >40× dari SRCNN. | Landasan arsitektur — menjelaskan asal-usul struktur 1×1/12-kanal yang jadi akar temuan AI-rendah di §1. |
| 5 | Abdollahzadeh, M. *Implementing Deep Convolutional Neural Networks in C without External Libraries* — [Medium/TDS](https://towardsdatascience.com/implementing-deep-convolutional-neural-networks-in-c-without-external-libraries-b30464f64d02/) & [GitHub: miladabd/DeepC](https://github.com/miladabd/DeepC) | Implementasi C murni (tanpa BLAS/pustaka eksternal) untuk inferensi FSRCNN pada video YUV 4:2:0. | **Basis kode** repo ini (`source.c`). Wajib dikutip sebagai *baseline implementation* yang dianalisis dan dioptimasi. |

### 3.3 Layout Memori (NCHW vs NHWC) & Optimisasi CPU — Pembanding Empiris

| # | Referensi | Temuan Utama | Relevansi dengan Tesis Ini |
|---|---|---|---|
| 6 | Georganas, E., Avancha, S., Banerjee, K., Kalamkar, D., Henry, G., Pabst, H., & Heinecke, A. (2018). *Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures*. SC18. [arXiv:1808.05567](https://arxiv.org/pdf/1808.05567) | Direct convolution pada CPU x86: urutan loop dan layout data menentukan efisiensi SIMD/reuse register. | Justifikasi teknis urutan loop *pixel-outer, channel-inner* pada `conv_layer_hwc` (bukan im2col). |
| 7 | de Prado, M., Mundy, A., Saeed, R., Denna, M., Pazos, N., & Benini, L. (2020). *Automated Design Space Exploration for Optimised Deployment of DNN on Arm Cortex-A CPUs*. [arXiv:2006.05181](https://arxiv.org/pdf/2006.05181) | NHWC mengungguli NCHW pada kernel 3×3 & 1×1 di Arm Cortex-A (latensi turun ~8%) — persis jenis kernel yang dipakai FSRCNN. | Pembanding hasil paling relevan (konteks CPU embedded, kernel identik dengan FSRCNN layer 2–7). |
| 8 | Lu, S., Chu, J., & Liu, X. T. (2022). *Im2win: Memory Efficient Convolution on SIMD Architectures*. HPEC 2022 / [arXiv:2306.14320](https://arxiv.org/html/2306.14320); lanjutan: *High Performance Im2win and Direct Convolutions using Three Tensor Layouts on SIMD Architectures*, [arXiv:2408.00278](https://arxiv.org/pdf/2408.00278) | NHWC pada skema im2win memberi speedup 11%–355% vs NCHW pada mesin SIMD, tergantung ukuran kernel/kanal. | Rentang speedup independen untuk pembanding hasil eksperimen tesis (≈12–19% pada mesin uji). |
| 9 | Liu, Y., Wang, Y., Yu, R., Li, M., Sharma, V., & Wang, Y. (2019). *Optimizing CNN Model Inference on CPUs*. USENIX ATC 2019. [PDF](https://www.usenix.org/system/files/atc19-liu-yizhi.pdf) | Manajemen layout data krusial untuk mengurangi overhead memori pada convolution CPU tanpa pustaka pihak ketiga. | Memperkuat bahwa optimisasi pola akses memori adalah kontribusi implementasi yang valid dan sudah terbukti pada skala produksi. |
| 10 | *Efficient Column-Wise N:M Pruning on RISC-V CPU*. [arXiv:2507.17301](https://arxiv.org/pdf/2507.17301) | Pada CPU RISC-V, layout non-standar (CNHW) bisa mengungguli NHWC baku hingga 1,8× untuk jaringan dangkal. | Catatan *limitations/future work*: layout optimal bergantung arsitektur CPU — HWC belum tentu optimal universal. |

## 4. Rekomendasi Judul Tesis

Judul-judul di bawah sengaja **tidak** memakai istilah "CHW ke HWC" sebagai frasa utama —
mengikuti masukan bahwa itu teknik, bukan kontribusi. Fokusnya dipindah ke *karakteristik
bandwidth memori* dan *pola akses/reuse data* sebagai objek penelitian.

| Opsi | Judul | Catatan |
|---|---|---|
| A (disarankan) | **"Karakterisasi Intensitas Aritmetika dan Optimalisasi Pola Akses Memori pada Inferensi FSRCNN di CPU Multi-Core untuk Super-Resolusi Video Real-Time"** | Memuat objek riset yang sebenarnya (intensitas aritmetika/roofline), bukan nama teknik implementasi. Selaras langsung dengan §1–2: bab hasil bisa dibuka dengan tabel AI per layer, lalu masuk ke teknik (reorganisasi pola akses + paralelisasi spasial) sebagai solusi yang diuji. |
| B | "Analisis Keterikatan Bandwidth Memori pada Lapisan Convolution Sempit (Pointwise) dan Implikasinya terhadap Paralelisasi FSRCNN di CPU" | Menonjolkan temuan spesifik paling menarik (layer 1×1 = titik AI terendah) sebagai fokus utama — cocok jika bab hasil ingin bertumpu pada temuan §1, bukan implementasi secara umum. |
| C | "Optimalisasi Reuse Data dan Granularitas Paralel pada Inferensi Model CNN Ringan (FSRCNN) di CPU Multi-Core" | Lebih generik/aman, menonjolkan dua kontribusi inti (#2 reuse data, #3 granularitas paralel) tanpa menyebut merek teknik atau nama model performa. |
| D | "Studi Roofline pada Inferensi CNN Ringan untuk Super-Resolusi Video Real-Time di CPU: Kasus FSRCNN" | Format "studi kasus" — memposisikan FSRCNN sebagai objek studi dari kerangka roofline yang lebih luas; cocok jika pembimbing ingin tesis dibingkai sebagai kontribusi metodologis (cara menganalisis) yang bisa digeneralisasi ke model lain, bukan cuma hasil optimisasi satu model. |

**Rekomendasi: Opsi A.** Judul ini menempatkan pertanyaan riset (kenapa & di mana FSRCNN
terikat bandwidth memori saat inferensi CPU) sebagai subjek utama, dan hasil optimisasi
sebagai jawabannya — bukan sebaliknya. Ini juga memberi ruang alami untuk bab tinjauan
pustaka mengikuti urutan §3.1 (teori) → §3.2 (arsitektur/baseline) → §3.3 (pembanding
empiris), dan bab hasil mengikuti urutan §1 (karakterisasi) → §2 (intervensi & dampaknya).
