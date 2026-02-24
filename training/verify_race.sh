#!/bin/bash

# Konfigurasi
ORIGINAL_WEIGHTS="weights_layer8_original.txt"
TRAINED_WEIGHTS="weights_layer8_trained.txt"
ACTIVE_WEIGHTS="weights_layer8.txt"
INPUT_YUV="tulips_yuv420_prog_planar_qcif.yuv"
GT_YUV="./ground_truth/ground_truth.yuv"
THREADS=8
RUNS=50
OUT_ORIGINAL="./results/psnr_original.txt"
OUT_TRAINED="./results/psnr_trained.txt"

# Reset data files
> "$OUT_ORIGINAL"
> "$OUT_TRAINED"

function run_test() {
    local label=$1
    local weight_file=$2
    local output_file=$3
    local total_psnr=0
    local min_psnr=999
    
    echo "---------------------------------------------------------"
    echo "Testing: $label ($weight_file)"
    
    # Pastikan file bobot yang benar digunakan
    cp "$weight_file" "$ACTIVE_WEIGHTS"
    
    for i in $(seq 1 $RUNS)
    do
        export OMP_NUM_THREADS=$THREADS
        ./fsrcnn_parallel "$INPUT_YUV" temp_out.yuv > /dev/null 2>&1
        
        # Ambil PSNR average
        psnr=$(ffmpeg -s 352x288 -i temp_out.yuv -s 352x288 -i "$GT_YUV" -lavfi psnr -f null - 2>&1 | grep -o "average:[0-9.]*" | cut -d: -f2)
        
        echo "  Run $i: $psnr dB"
        echo "$psnr" >> "$output_file"
        
        total_psnr=$(echo "$total_psnr + $psnr" | bc)
        if (( $(echo "$psnr < $min_psnr" | bc -l) )); then
            min_psnr=$psnr
        fi
    done
    
    avg_psnr=$(echo "scale=4; $total_psnr / $RUNS" | bc)
    echo ">> Result for $label: Avg = $avg_psnr dB, Min = $min_psnr dB"
    echo ""
}

# --- Main Execution ---
echo "========================================================="
echo "FSRCNN RACE CONDITION COMPENSATION COMPARISON"
echo "========================================================="

# 1. Cek keberanian file
if [[ ! -f "$ORIGINAL_WEIGHTS" || ! -f "$TRAINED_WEIGHTS" ]]; then
    echo "Error: Pastikan $ORIGINAL_WEIGHTS dan $TRAINED_WEIGHTS ada di folder ini."
    exit 1
fi

# 2. Jalankan Uji
run_test "BEFORE TRAINING (Original)" "$ORIGINAL_WEIGHTS" "$OUT_ORIGINAL"
run_test "AFTER TRAINING (Compensated)" "$TRAINED_WEIGHTS" "$OUT_TRAINED"

# 3. Generate Chart
if [ -d "venv" ]; then
    echo "========================================================="
    echo "Generating Stability Chart..."
    source venv/bin/activate
    python3 chart.py
    deactivate
    echo "Chart generated: psnr_stability_analysis.png"
else
    echo "Warning: venv not found. Skipping chart generation."
fi

# Bersihkan
rm temp_out.yuv
echo "========================================================="
echo "Selesai! Bandingkan nilai 'Min PSNR' di atas."