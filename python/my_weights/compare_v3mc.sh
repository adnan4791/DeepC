#!/bin/bash

# ===========================================================
# FSRCNN v3 + Micro-Compensator: PSNR Comparison
# ===========================================================
# Membandingkan: Original vs v3 vs v3+MicroComp

# Konfigurasi
EXEC_OLD="./fsrcnn_parallel_old"            # bobot original
EXEC_V3="./fsrcnn_parallel_layer8v3"        # v3 (channel equalization)
EXEC_V3MC="./fsrcnn_v3_micro_comp"          # v3 + micro-compensator
INPUT_YUV="suzie_qcif.yuv"
GT_YUV="clean.yuv"
THREADS=${1:-8}                              # default 8, bisa diubah: ./compare_v3mc.sh 6
RUNS=10

OUT_OLD="psnr_old.txt"
OUT_V3="psnr_v3.txt"
OUT_V3MC="psnr_v3mc.txt"

# Resolusi output HR
WIDTH=352
HEIGHT=288
PIX_FMT=yuv420p

# ---------- Cek file ----------
for f in "$EXEC_OLD" "$EXEC_V3" "$EXEC_V3MC" "$INPUT_YUV" "$GT_YUV"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ File tidak ditemukan: $f"
        exit 1
    fi
done

> "$OUT_OLD"
> "$OUT_V3"
> "$OUT_V3MC"

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

        psnr=$(ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt $PIX_FMT -i temp_out.yuv \
                      -s ${WIDTH}x${HEIGHT} -pix_fmt $PIX_FMT -i "$GT_YUV" \
                      -lavfi psnr -f null - 2>&1 \
               | grep -o "average:[0-9.inf]*" | cut -d: -f2)

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
echo "  FSRCNN: ORIGINAL vs V3 vs V3+MICRO-COMPENSATOR"
echo "========================================================="
echo "  Input    : $INPUT_YUV"
echo "  GT       : $GT_YUV"
echo "  Threads  : $THREADS"
echo "  Runs     : $RUNS"
echo "========================================================="

run_test "ORIGINAL (bobot lama)" "$EXEC_OLD" "$OUT_OLD"
run_test "V3 CHANNEL EQUALIZATION (layer 8 only)" "$EXEC_V3" "$OUT_V3"
run_test "V3 + MICRO-COMPENSATOR (v3 + 8-feat CNN)" "$EXEC_V3MC" "$OUT_V3MC"

# Bersihkan
rm -f temp_out.yuv

echo "========================================================="
echo "  RINGKASAN"
echo "========================================================="
echo "  Hasil PSNR original    : $OUT_OLD"
echo "  Hasil PSNR v3          : $OUT_V3"
echo "  Hasil PSNR v3+microcomp: $OUT_V3MC"
echo "========================================================="
echo "Selesai!"
