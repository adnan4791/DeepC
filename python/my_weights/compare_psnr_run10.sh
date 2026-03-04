#!/bin/bash

# ===========================================================
# FSRCNN RACE CONDITION: Perbandingan Bobot Original vs Finetuned
# ===========================================================
# Script ini menjalankan executable dengan bobot original dan finetuned
# masing-masing N kali, lalu membandingkan PSNR terhadap ground truth.

# Konfigurasi
EXEC_OLD="./fsrcnn_parallel_old"      # executable dengan bobot original (semua layer)
# EXEC_NEW="./fsrcnn_parallel"           # executable dengan bobot finetuned (semua layer)
EXEC_L8="./fsrcnn_parallel_layer8v4"     # executable dengan bobot finetuned (layer 8 saja)
INPUT_YUV="suzie_qcif.yuv"
GT_YUV="clean.yuv"                     # ground truth (dari critical/serial)
THREADS=2
RUNS=10
OUT_ORIGINAL="psnr_old.txt"
OUT_FINETUNED="psnr_new.txt"
OUT_LAYER8="psnr_layer8v4.txt"

# Resolusi output HR
WIDTH=352
HEIGHT=288
PIX_FMT=yuv420p

# ---------- Cek file ----------
for f in "$EXEC_OLD" "$EXEC_L8" "$INPUT_YUV" "$GT_YUV"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ File tidak ditemukan: $f"
        exit 1
    fi
done

# Reset data files
> "$OUT_ORIGINAL"
> "$OUT_FINETUNED"
> "$OUT_LAYER8"

# ---------- Fungsi run_test ----------
function run_test() {
    local label=$1
    local executable=$2
    local output_file=$3
    local total_psnr=0
    local min_psnr=999
    local max_psnr=0

    echo "---------------------------------------------------------"
    echo "Testing: $label"
    echo "  Executable: $executable"
    echo "  Runs: $RUNS, Threads: $THREADS"

    for i in $(seq 1 $RUNS)
    do
        export OMP_NUM_THREADS=$THREADS
        $executable "$INPUT_YUV" temp_out.yuv > /dev/null 2>&1

        # Hitung PSNR average menggunakan ffmpeg
        psnr=$(ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt $PIX_FMT -i temp_out.yuv \
                      -s ${WIDTH}x${HEIGHT} -pix_fmt $PIX_FMT -i "$GT_YUV" \
                      -lavfi psnr -f null - 2>&1 \
               | grep -o "average:[0-9.inf]*" | cut -d: -f2)

        # Fallback jika psnr kosong
        if [[ -z "$psnr" ]]; then
            echo "  Run $i: FAILED"
            psnr=0
        else
            echo "  Run $i: $psnr dB"
        fi

        echo "$psnr" >> "$output_file"

        total_psnr=$(echo "$total_psnr + $psnr" | bc)
        if (( $(echo "$psnr < $min_psnr" | bc -l) )); then
            min_psnr=$psnr
        fi
        if (( $(echo "$psnr > $max_psnr" | bc -l) )); then
            max_psnr=$psnr
        fi
    done

    avg_psnr=$(echo "scale=4; $total_psnr / $RUNS" | bc)
    echo ""
    echo ">> $label:"
    echo "   Avg = $avg_psnr dB"
    echo "   Min = $min_psnr dB"
    echo "   Max = $max_psnr dB"
    echo ""
}

# ---------- Main ----------
echo "========================================================="
echo "  FSRCNN RACE CONDITION: ORIGINAL vs FINETUNED"
echo "========================================================="
echo "  Input    : $INPUT_YUV"
echo "  GT       : $GT_YUV"
echo "  Threads  : $THREADS"
echo "  Runs     : $RUNS"
echo "========================================================="

run_test "ORIGINAL (bobot lama)" "$EXEC_OLD" "$OUT_ORIGINAL"
# run_test "FINETUNED ALL LAYER (bobot baru semua layer)" "$EXEC_NEW" "$OUT_FINETUNED"
run_test "FINETUNED LAYER 8 ONLY (bobot baru layer 8 saja)" "$EXEC_L8" "$OUT_LAYER8"

# Bersihkan
rm -f temp_out.yuv

echo "========================================================="
echo "  RINGKASAN"
echo "========================================================="
echo "  Hasil PSNR original disimpan di: $OUT_ORIGINAL"
echo "  Hasil PSNR finetuned (all)  di: $OUT_FINETUNED"
echo "  Hasil PSNR finetuned (L8)   di: $OUT_LAYER8"
echo "========================================================="
echo "Selesai!"
