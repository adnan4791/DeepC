# Dokumentasi Implementasi FSRCNN (Python Version)

Dokumentasi ini menjelaskan detail teknis konversi algoritma **Fast Super-Resolution Convolutional Neural Network (FSRCNN)** dari bahasa C ke Python menggunakan framework PyTorch.

## 1. Arsitektur Model (8-Layer)

Model ini mengikuti struktur FSRCNN standar yang terdiri dari empat bagian utama:

| Bagian | Nama Layer | Fungsi | Konfigurasi |
| --- | --- | --- | --- |
| **Feature Extraction** | Layer 1 | Ekstraksi fitur dari input luma (Y). | Kernel 5x5, 56 filters |
| **Shrinking** | Layer 2 | Mengurangi dimensi fitur untuk efisiensi. | Kernel 1x1, 12 filters |
| **Non-linear Mapping** | Layer 3 - 6 | Melakukan pemetaan non-linear (m=4). | Kernel 3x3, 12 filters |
| **Expanding** | Layer 7 | Mengembalikan dimensi fitur ke ukuran awal. | Kernel 1x1, 56 filters |
| **Deconvolution** | Layer 8 | Melakukan perbesaran (upscaling) gambar. | Kernel 9x9, Stride=Scale |

## 2. Detail Implementasi PReLU

Berbeda dengan model deep learning modern yang melatih koefisien PReLU, kode ini menggunakan koefisien **statis (fixed)** yang diambil langsung dari *hardcoded value* pada kode C:

* **Layer 1:** `-0.8986`
* **Layer 2:** `0.3236`
* **Layer 3:** `0.2288`
* **Layer 4:** `0.2476`
* **Layer 5:** `0.3495`
* **Layer 6:** `0.7806`
* **Layer 7:** `0.0087`

## 3. Mekanisme Loading Bobot (C Compatibility)

Karena adanya perbedaan antara cara C membaca file (sequential pointer) dan Python (array-based), script ini menggunakan fungsi `load_txt_precise`:

* **Slicing Data:** Menggunakan parameter `count` untuk memotong data `.txt` yang berlebih. Contoh: Jika `weights_layer2.txt` berisi 1400 angka namun model hanya butuh 672, script hanya akan mengambil 672 angka pertama (meniru `fscanf` di C).
* **Reshaping:** Data linear dari teks diubah menjadi tensor 4D dengan urutan `[Out_Channels, In_Channels, Height, Width]`.

## 4. Pemrosesan Video YUV 4:2:0

Pemrosesan dilakukan per frame dengan alur sebagai berikut:

### Komponen Y (Luma)

* Diekstraksi sebesar `width * height` bytes.
* Dinormalisasi ke rentang $[0, 1]$ (atau tetap $0-255$ tergantung konfigurasi training).
* Diproses melalui Neural Network di GPU/CPU.

### Komponen U & V (Chroma)

* Sesuai dengan kode C asli, komponen warna tidak diproses oleh AI.
* Menggunakan metode **Simple Repetition**: Setiap 1 pixel warna diulang menjadi blok $2 \times 2$ untuk menyesuaikan skala $2\times$ (Upscale).
* Ini memastikan warna tetap sinkron dengan kecerahan (Luma).

## 5. Perbedaan Utama dengan Versi C

1. **Presisi:** Python menggunakan `float32` secara default, sementara C menggunakan `double` (`float64`).
2. **Kecepatan:** Versi Python mendukung akselerasi GPU (CUDA), sedangkan versi C menggunakan OpenMP untuk CPU multi-threading.
3. **Output:** Penulisan frame di Python menggunakan buffer stream yang lebih dinamis dibandingkan alokasi statis di C.

---

### Cara Menjalankan

1. Pastikan file bobot ada di folder `./my_weights`.
2. Jalankan script:
```bash
python3 newversion.py

```


3. Cek hasil menggunakan FFplay:
```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 352x288 output_final.yuv

```
