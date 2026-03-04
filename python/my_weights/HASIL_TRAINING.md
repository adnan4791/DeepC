# Dokumentasi Hasil Fine-Tuning FSRCNN
## Noise-Aware Training untuk Race Condition Robustness

**Tanggal**: 4 Maret 2026  
**Script**: `python/my_weights/finetunning.py`  
**Bobot Original**: `fsrcnn_original.pth`

---

## 1. Latar Belakang

FSRCNN Layer 8 menggunakan `#pragma omp parallel for` pada loop deconvolution, di mana 56 channel diakumulasi via `imadd()` ke buffer yang sama **tanpa** `#pragma omp critical`. Hal ini menyebabkan **race condition** yang membuat output non-deterministik.

Eksperimen 30 kali run menunjukkan:
- **Modus** (nilai paling sering) dari output race condition = nilai deterministik (dengan critical)
- Std. deviasi noise ≈ 0.012
- Noise bersifat additive, centered di sekitar 0

**Tujuan**: Fine-tune bobot agar model tetap menghasilkan output optimal meskipun tanpa `#pragma omp critical` (menghindari lock contention yang memperlambat eksekusi).

---

## 2. Metodologi

### Noise-Aware Training
Saat training, noise empiris yang menyimulasikan race condition diinjeksikan ke output deconvolution layer:

```
output = FSRCNN(input_LR) + noise_race_condition
loss = L1(output, ground_truth)
```

### Sumber Noise
Noise dihitung dari data training yang dihasilkan oleh C executable:
- **Ground Truth**: `train_data/gt/frame_XXXX.yuv` (150 frame, deterministik dengan `omp_set_num_threads(1)`)
- **Race Condition**: `train_data/run0..29/frame_XXXX.yuv` (30 run paralel tanpa critical)
- **Noise** = `runX - gt` (per piksel)

### Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Input LR | suzie_qcif.yuv (176×144, Y-only) |
| Output HR | 352×288 |
| Scale | 2× |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Loss Function | L1 (robust terhadap outlier) |
| Batch Size | 4 |
| Epoch | 30 |
| Noise Probability | 70% per batch |
| Noise Pool | 27 sampel empiris dari train_data/ |
| Device | CPU |

---

## 3. Hasil Training

### Progres PSNR per Epoch

| Epoch | Loss | PSNR (clean) | PSNR (noisy) |
|---|---|---|---|
| Baseline | — | 38.39 dB | 34.62 dB |
| 1 | 0.006814 | 39.76 dB | 35.48 dB |
| 5 | 0.005747 | 42.90 dB | 36.83 dB |
| 10 | 0.005642 | 45.03 dB | 37.74 dB |
| 15 | 0.004422 | 46.88 dB | 40.04 dB |
| 20 | 0.004114 | 48.43 dB | 39.53 dB |
| 25 | 0.004243 | 49.50 dB | 40.09 dB |
| **30** | **0.003555** | **50.23 dB** | **40.57 dB** |

### Evaluasi Akhir (Model Terbaik, Loss = 0.003385)

| Metrik | Sebelum Fine-Tuning | Sesudah Fine-Tuning | Peningkatan |
|---|---|---|---|
| PSNR (tanpa noise) | 38.39 dB | 47.21 dB | **+8.82 dB** |
| PSNR (dengan noise RC) | 34.62 dB | 39.56 dB | **+4.94 dB** |

---

## 4. File Output

### Model PyTorch
- `fsrcnn_finetuned.pth` — model terbaik (loss terendah)
- `fsrcnn_finetuned_epoch{5,10,15,20,25,30}.pth` — checkpoint per 5 epoch

### Bobot untuk C Code (format .txt)
| File | Jumlah Nilai |
|---|---|
| `weights_layer1_finetuned.txt` | 1400 |
| `biasess_layer1_finetuned.txt` | 56 |
| `weights_layer2_finetuned.txt` | 672 |
| `biasess_layer2_finetuned.txt` | 12 |
| `weights_layer3_finetuned.txt` | 1296 |
| `biasess_layer3_finetuned.txt` | 12 |
| `weights_layer4_finetuned.txt` | 1296 |
| `biasess_layer4_finetuned.txt` | 12 |
| `weights_layer5_finetuned.txt` | 1296 |
| `biasess_layer5_finetuned.txt` | 12 |
| `weights_layer6_finetuned.txt` | 1296 |
| `biasess_layer6_finetuned.txt` | 12 |
| `weights_layer7_finetuned.txt` | 672 |
| `biasess_layer7_finetuned.txt` | 56 |
| `weights_layer8_finetuned.txt` | 4536 |
| `biasess_layer8_finetuned.txt` | 1 |

---

## 5. Cara Penggunaan di C Code

1. Copy file `*_finetuned.txt` ke folder C executable
2. Ubah `fopen()` pada C code agar membaca file `*_finetuned.txt`, atau rename file menjadi nama originalnya
3. Compile tanpa perubahan arsitektur (jumlah layer & filter tetap sama)
4. Jalankan **tanpa** `#pragma omp critical` di Layer 8

---

## 6. Arsitektur Model (Tidak Berubah)

```
Layer 1: Conv2d(1→56, 5×5)   + PReLU(α=-0.8986)   [Feature Extraction]
Layer 2: Conv2d(56→12, 1×1)  + PReLU(α=0.3236)     [Shrinking]
Layer 3: Conv2d(12→12, 3×3)  + PReLU(α=0.2288)     [Mapping]
Layer 4: Conv2d(12→12, 3×3)  + PReLU(α=0.2476)     [Mapping]
Layer 5: Conv2d(12→12, 3×3)  + PReLU(α=0.3495)     [Mapping]
Layer 6: Conv2d(12→12, 3×3)  + PReLU(α=0.7806)     [Mapping]
Layer 7: Conv2d(12→56, 1×1)  + PReLU(α=0.0087)     [Expanding]
Layer 8: DeConv(56→1, 9×9, stride=2)                [Upsampling] ← race condition di sini
```

---

## 7. Catatan

- Noise pool hanya berisi 27 sampel (dari 4500 potensial) karena sebagian besar perbedaan runX vs GT berada di bawah threshold uint8 (< 0.5/255)
- PReLU koefisien bersifat **tetap** (tidak trainable), sesuai implementasi C
- Training dilakukan pada full frame (tanpa patch cropping) karena resolusi sudah kecil (144×176 LR)
