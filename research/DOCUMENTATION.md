# Dokumentasi Kode Implementasi FSRCNN (Fast Super-Resolution Convolutional Neural Network)

Dokumen ini menjelaskan implementasi algoritma **FSRCNN** untuk interpolasi video format YUV 4:2:0 yang terdapat dalam file `source.c`. Kode ini dikembangkan oleh Milad Abdollahzadeh pada 09/02/2017.

## Daftar Isi
1.  [Gambaran Umum](#gambaran-umum)
2.  [Persyaratan Sistem dan Kompilasi](#persyaratan-sistem-dan-kompilasi)
3.  [Cara Penggunaan](#cara-penggunaan)
4.  [Arsitektur Program](#arsitektur-program)
    *   [Struktur Data Utama](#struktur-data-utama)
    *   [Fungsi Utama (main)](#fungsi-utama-main)
    *   [Jaringan Saraf Tiruan (FSRCNN)](#jaringan-saraf-tiruan-fsrcnn)
5.  [Detail Fungsi-Fungsi](#detail-fungsi-fungsi)
6.  [Keterbatasan Implementasi Saat Ini](#keterbatasan-implementasi-saat-ini)

---

## Gambaran Umum

Program ini dirancang untuk melakukan *upscaling* (peningkatan resolusi) video input dengan faktor skala **2x** (contoh: QCIF 176x144 menjadi CIF 352x288).

**Metode Pengolahan:**
-   **Komponen Y (Luminance):** Diolah menggunakan jaringan saraf tiruan dalam (Deep Neural Network) untuk mendapatkan detail resolusi tinggi yang akurat.
-   **Komponen U dan V (Chrominance):** Diolah menggunakan metode replikasi piksel sederhana (*Nearest Neighbor Upsampling*) karena mata manusia kurang sensitif terhadap detail warna dibandingkan detail kecerahan.

## Persyaratan Sistem dan Kompilasi

### Ketergantungan (Dependencies)
-   **Compiler C**: GCC atau kompatibel.
-   **OpenMP**: Digunakan untuk memparalelkan proses konvolusi (`#pragma omp parallel for`).
-   **Library Standar**: `stdio.h`, `stdlib.h`.
-   **File Bobot (Weights)**: Program membutuhkan file eksternal `.txt` berisi bobot jaringan yang telah dilatih sebelumnya.

### Cara Kompilasi
Untuk mengkompilasi program, gunakan perintah berikut di terminal:

```bash
```bash
gcc-15 source.c -o apps/fsrcnn -fopenmp -lm
```
*(Flag `-fopenmp` wajib disertakan untuk mengaktifkan pemrosesan paralel dan kinerja optimal)*

## Cara Penggunaan

Program dijalankan melalui baris perintah (command line) dengan menyertakan nama file input dan output.

**Sintaks:**
```bash
./apps/fsrcnn <file_input> <file_output>
```

**Argumen:**
1.  `<file_input>`: Path ke file video RAW format YUV 4:2:0.
2.  `<file_output>`: Path tujuan untuk menyimpan video hasil upscaling.

**Contoh:**
```bash
./apps/fsrcnn input_qcif.yuv output_cif.yuv
```

## Visualisasi Video (FFmpeg/FFplay)

Karena format video adalah **RAW YUV 4:2:0** (tanpa header), video player biasa mungkin tidak bisa memutarnya secara langsung. Gunakan `ffplay` (bagian dari FFmpeg) dengan menentukan format dan resolusi secara eksplisit.

**Perintah Dasar:**
```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size <LEBAR>x<TINGGI> <nama_file.yuv>
```

**Contoh untuk Input (QCIF 176x144):**
```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 176x144 input_qcif.yuv
```

**Contoh untuk Output (CIF 352x288 - Hasil Upscale 2x):**
```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 352x288 output_cif.yuv
```

## Arsitektur Program

### Struktur Data Utama
Program menggunakan array global untuk menyimpan bobot dan bias jaringan yang dimuat dari file teks eksternal:
-   `weights_layerX`: Array bobot untuk setiap layer (1-8), disimpan dalam folder `weights/`.
-   `biases_layerX`: Array bias untuk setiap layer (1-8), disimpan dalam folder `weights/`.
-   Buffer gambar: `inBuf` (input), `outBuf` (output), `img_hr` (high-res), `img_lr` (low-res).

### Fungsi Utama (main)
Alur eksekusi utama adalah sebagai berikut:
1.  **Inisialisasi**: Menetapkan parameter video (Resolusi Input: 176x144, Jumlah Frame: 150, Skala: 2).
2.  **Validasi File**: Membuka dan mengecek ketersediaan file input, output, dan seluruh file bobot di folder `weights/` (`weights_layer1.txt`, dst).
3.  **Loop Pemrosesan Frame**:
    -   Membaca frame Y, U, dan V.
    -   **Kanal Y**: Dinormalisasi ke [0,1], diproses oleh fungsi `FSRCNN()`, lalu didenormalisasi kembali ke [0,255].
    -   **Kanal U & V**: Di-upscale menggunakan replikasi piksel (tiap 1 piksel menjadi blok 2x2).
    -   Menulis frame hasil ke file output.

### Jaringan Saraf Tiruan (FSRCNN)
Fungsi `FSRCNN` mengimplementasikan jaringan saraf 8 layer dengan struktur:

| Layer | Tipe | Filter Size | Jumlah Filter | Fungsi Aktivasi | Keterangan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Konvolusi | 5x5 | 56 | PReLU (-0.8986) | Ekstraksi Fitur |
| **2** | Konvolusi | 1x1 | 12 | PReLU (0.3236) | Shrinking / Mapping |
| **3-6** | Konvolusi | 3x3 | 12 | PReLU (Variatif) | Non-linear Mapping |
| **7** | Konvolusi | 1x1 | 56 | PReLU (0.0087) | Expanding |
| **8** | Dekonvolusi | 9x9 | 1 | - | Rekonstruksi HR |

## Detail Fungsi-Fungsi

Berikut adalah penjelasan fungsi-fungsi pendukung dalam kode `source.c`:

### 1. `imfilter`
Melakukan operasi konvolusi 2D standar.
-   Input: Citra asli, kernel konvolusi.
-   Proses: Melakukan *padding* terlebih dahulu menggunakan `pad_image`, kemudian menggeser kernel ke seluruh citra untuk menghitung nilai piksel baru.
-   Output: Citra hasil filtering (`img_fltr`).

### 2. `pad_image`
Menambahkan padding di sekeliling citra untuk memproses tepian gambar.
-   **Jenis Padding**: Replikasi (nilai piksel tepi disalin ke area padding).
-   Menangani 4 sisi (atas, bawah, kiri, kanan) dan 4 sudut secara eksplisit.

### 3. `PReLU` (Parametric Rectified Linear Unit)
Fungsi aktivasi non-linier yang digunakan setelah setiap lapisan konvolusi (kecuali layer terakhir).
-   Rumus: $f(x) = \begin{cases} x, & \text{if } x > 0 \\ a \cdot x, & \text{if } x \le 0 \end{cases}$
-   Nilai koefisien $a$ (`prelu_coeff`) berbeda untuk setiap layer dan telah dilatih sebelumnya.

### 4. `imadd`
Menambahkan hasil konvolusi ke akumulator (feature map).
-   Digunakan untuk menggabungkan hasil dari banyak filter input ke satu feature map output.

### 5. `deconv`
Melakukan operasi dekonvolusi (transposed convolution) untuk lapisan terakhir.
-   Berfungsi untuk memetakan fitur berdimensi rendah kembali ke citra resolusi tinggi.
-   Menggunakan *stride* untuk upsampling spasial.

### 6. `double_2_uint8`
Mengonversi data tipe `double` (hasil perhitungan neural network) menjadi `unsigned char` (format pixel 8-bit).
-   Melakukan *clamping*: Nilai < 0 menjadi 0, > 255 menjadi 255.
-   Melakukan pembulatan nilai desimal ke integer terdekat.

## Keterbatasan Implementasi Saat Ini

1.  **Resolusi Statis**: Dimensi video input (176x144) dan jumlah frame (150) di-*hardcode* di dalam fungsi `main`. Perubahan resolusi memerlukan rekompilasi ulang.
2.  **Dependensi File Eksternal**: Membutuhkan 15+ file teks berisi bobot (`weights/weights_*.txt`, `weights/biases_*.txt`) relatif terhadap direktori eksekusi.
3.  **Penanganan Kesalahan (Error Handling)**: Program akan mencetak pesan error jika file bobot tidak ditemukan, namun di beberapa bagian eksekusi tetap berlanjut yang dapat menyebabkan *segmentation fault*.
4.  **Dowload Assets YUV Video**: File YUV video bisa diunduh manual [disini](https://web.archive.org/web/20190220164028/http://www.sunrayimage.com/examples.html).
