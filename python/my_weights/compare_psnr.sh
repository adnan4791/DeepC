#!/bin/bash
# ===========================================================
# Perbandingan PSNR: output_old vs output_new terhadap clean.yuv (GT)
# ===========================================================
# Cara pakai:
#   ./compare_psnr.sh <output_old.yuv> <output_new.yuv> <clean.yuv>
# Contoh:
#   ./compare_psnr.sh noisy1.yuv clean1.yuv clean1.yuv
#   ./compare_psnr.sh output_original_1.yuv output_trained_1.yuv clean1.yuv

# Resolusi output (HR) — sesuaikan jika berbeda
WIDTH=352
HEIGHT=288
PIX_FMT=yuv420p

# ---------- Cek argumen ----------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <output_old.yuv> <output_new.yuv> <ground_truth.yuv>"
    echo ""
    echo "Contoh: $0 noisy1.yuv clean1.yuv clean1.yuv"
    exit 1
fi

OUTPUT_OLD="$1"
OUTPUT_NEW="$2"
GT_YUV="$3"

for f in "$OUTPUT_OLD" "$OUTPUT_NEW" "$GT_YUV"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ File tidak ditemukan: $f"
        exit 1
    fi
done

# ---------- Fungsi hitung PSNR ----------
calc_psnr() {
    local input="$1"
    local gt="$2"
    local label="$3"

    result=$(ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt $PIX_FMT -i "$input" \
                    -s ${WIDTH}x${HEIGHT} -pix_fmt $PIX_FMT -i "$gt" \
                    -lavfi psnr -f null - 2>&1)

    psnr=$(echo "$result" | grep -o "average:[0-9.inf]*" | cut -d: -f2)

    if [[ -z "$psnr" ]]; then
        echo "  $label: GAGAL menghitung PSNR"
        echo "  Output ffmpeg:"
        echo "$result" | tail -3
    else
        echo "  $label: $psnr dB"
    fi
}

# ---------- Main ----------
echo "========================================================="
echo "  PERBANDINGAN PSNR"
echo "========================================================="
echo "  Ground Truth : $GT_YUV"
echo "  Output OLD   : $OUTPUT_OLD"
echo "  Output NEW   : $OUTPUT_NEW"
echo "  Resolusi     : ${WIDTH}x${HEIGHT} ($PIX_FMT)"
echo "---------------------------------------------------------"

calc_psnr "$OUTPUT_OLD" "$GT_YUV" "OLD (original weights)"
calc_psnr "$OUTPUT_NEW" "$GT_YUV" "NEW (finetuned weights)"

echo "========================================================="
echo "Selesai!"
