#!/bin/bash

# Konfigurasi
INPUT="suzie_qcif.yuv"
GT="suzie_qcif_serial_hr.yuv"
BINARIES=("fsrcnn_selectcpu" "fsrcnn_syncpilot_selectcpu")
WIDTH=352
HEIGHT=288
FORMAT="yuv420p"

echo "=========================================================="
echo "      SCRIPT KOMPARASI KECEPATAN & KUALITAS FSRCNN        "
echo "=========================================================="
echo "Input          : $INPUT"
echo "Ground Truth   : $GT"
echo "Resolusi Output: ${WIDTH}x${HEIGHT}"
echo "=========================================================="

# Cek apakah file input dan GT ada
if [ ! -f "$INPUT" ]; then
    echo "Error: File input $INPUT tidak ditemukan!"
    exit 1
fi

if [ ! -f "$GT" ]; then
    echo "Error: File ground truth $GT tidak ditemukan!"
    exit 1
fi

# Looping untuk setiap binary
RUNS=5

for BIN in "${BINARIES[@]}"; do
    if [ ! -f "$BIN" ]; then
        echo "Peringatan: Binary $BIN tidak ditemukan, dilewati."
        echo "----------------------------------------------------------"
        continue
    fi

    OUT_YUV="out_${BIN}.yuv"
    LOG_FFMPEG="log_ffmpeg_${BIN}.txt"

    echo ">> Menjalankan $BIN sebanyak $RUNS kali ..."
    
    SUM_TIME=0
    SUM_PSNR=0
    SUM_SSIM=0
    VALID_RUNS=0

    for ((i=1; i<=RUNS; i++)); do
        echo -ne "   Run $i/$RUNS ... "
        
        # Eksekusi dan hitung waktu (menggunakan python agar mendapat presisi desimal detik)
        START_TIME=$(python3 -c 'import time; print(time.time())')
        ./$BIN 4 8 "$INPUT" "$OUT_YUV" >/dev/null 2>&1
        END_TIME=$(python3 -c 'import time; print(time.time())')
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc -l)

        # Cek PSNR & SSIM
        if [ -f "$OUT_YUV" ]; then
            ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt $FORMAT -i "$OUT_YUV" \
                   -s ${WIDTH}x${HEIGHT} -pix_fmt $FORMAT -i "$GT" \
                   -lavfi "[0:v][1:v]psnr;[0:v][1:v]ssim" -f null - > "$LOG_FFMPEG" 2>&1

            
            PSNR=$(grep "Parsed_psnr" "$LOG_FFMPEG" | grep -o "average:[0-9.]*" | cut -d':' -f2 | tail -n 1)
            SSIM=$(grep "Parsed_ssim" "$LOG_FFMPEG" | grep -o "All:[0-9.]*" | cut -d':' -f2 | tail -n 1)
            
            if [ -z "$PSNR" ]; then PSNR="0"; fi
            if [ -z "$SSIM" ]; then SSIM="0"; fi

            # Tampilkan waktu execution dengan format 2 desimal biar rapi
            ELAPSED_FMT=$(printf "%.2f" $ELAPSED)
            echo "Waktu: ${ELAPSED_FMT}s | PSNR: $PSNR dB | SSIM: $SSIM"

            # Akumulasi untuk rata-rata
            SUM_TIME=$(echo "$SUM_TIME + $ELAPSED" | bc -l)
            SUM_PSNR=$(echo "$SUM_PSNR + $PSNR" | bc -l)
            SUM_SSIM=$(echo "$SUM_SSIM + $SSIM" | bc -l)
            VALID_RUNS=$((VALID_RUNS + 1))
        else
            echo "Gagal: $OUT_YUV tidak terbentuk!"
        fi
    done

    if [ $VALID_RUNS -gt 0 ]; then
        AVG_TIME=$(echo "scale=3; $SUM_TIME / $VALID_RUNS" | bc -l)
        AVG_PSNR=$(echo "scale=4; $SUM_PSNR / $VALID_RUNS" | bc -l)
        AVG_SSIM=$(echo "scale=4; $SUM_SSIM / $VALID_RUNS" | bc -l)
        
        echo "   ========================================="
        echo "   RATA-RATA DARI $VALID_RUNS RUN:"
        echo "   Waktu Eksekusi : $AVG_TIME detik"
        echo "   PSNR           : $AVG_PSNR dB"
        echo "   SSIM           : $AVG_SSIM"
    fi
    echo "----------------------------------------------------------"
done

echo "Komparasi selesai."
