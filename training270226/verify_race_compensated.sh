#!/bin/bash

# Konfigurasi
ORIGINAL_WEIGHTS="weights_layer8_original.txt"
TRAINED_WEIGHTS="weights_layer8_trained.txt"
ACTIVE_WEIGHTS="weights_layer8.txt"
ORIGINAL_BIASESS="biasess_layer8_original.txt"
TRAINED_BIASESS="biasess_layer8_trained.txt"
ACTIVE_BIASESS="biasess_layer8.txt"
INPUT_YUV="suzie_qcif.yuv"
GT_YUV="./ground_truth/suzie_qcif_serial_2.yuv"
THREADS=8
RUNS=5
OUT_ORIGINAL_150="./results/psnr_150_original.txt"
OUT_TRAINED_150="./results/psnr_150_trained.txt"
OUT_ORIGINAL_ORDERED="./results/psnr_ordered_original.txt"
OUT_TRAINED_ORDERED="./results/psnr_ordered_trained.txt"

# Reset data files
> "$OUT_ORIGINAL_150"
> "$OUT_TRAINED_150"
> "$OUT_ORIGINAL_ORDERED"
> "$OUT_TRAINED_ORDERED"

function run_test() {
    local label=$1
    local weight_file=$2
    local biases_file=$3
    local output_file=$4
    local binary=$5
    local total_psnr=0
    local min_psnr=999
    
    echo "---------------------------------------------------------"
    echo "Testing: $label ($weight_file)"
    echo "Testing: $label ($biases_file)"
    
    # Pastikan file bobot yang benar digunakan
    cp "$weight_file" "$ACTIVE_WEIGHTS"
    cp "$biases_file" "$ACTIVE_BIASESS"
    
    for i in $(seq 1 $RUNS)
    do
        export OMP_NUM_THREADS=$THREADS
        $binary "$INPUT_YUV" temp_out.yuv > /dev/null 2>&1
        
        # Ambil PSNR average (need -pix_fmt yuv420p for raw YUV files)
        ffmpeg_output=$(ffmpeg -s 352x288 -pix_fmt yuv420p -i temp_out.yuv -s 352x288 -pix_fmt yuv420p -i "$GT_YUV" -lavfi psnr -f null - 2>&1)
        psnr=$(echo "$ffmpeg_output" | grep -o "average:[0-9.inf]*" | cut -d: -f2)
        
        # Fallback jika psnr kosong
        if [[ -z "$psnr" ]]; then
            echo "  Run $i: FAILED (ffmpeg output below)"
            echo "$ffmpeg_output" | tail -5
            psnr=0
        else
            echo "  Run $i: $psnr dB"
        fi
        
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
echo "--- Uji menggunakan binari fsrcnn_parallel_150 ---"
run_test "BEFORE TRAINING (Original) - 150" "$ORIGINAL_WEIGHTS" "$ORIGINAL_BIASESS" "$OUT_ORIGINAL_150" "./fsrcnn_parallel_150"
run_test "AFTER TRAINING (Compensated) - 150" "$TRAINED_WEIGHTS" "$TRAINED_BIASESS" "$OUT_TRAINED_150" "./fsrcnn_parallel_150"

echo "--- Uji menggunakan binari fsrcnn_parallel_ordered_training ---"
run_test "BEFORE TRAINING (Original) - Ordered" "$ORIGINAL_WEIGHTS" "$ORIGINAL_BIASESS" "$OUT_ORIGINAL_ORDERED" "./fsrcnn_parallel_ordered_training"
run_test "AFTER TRAINING (Compensated) - Ordered" "$TRAINED_WEIGHTS" "$TRAINED_BIASESS" "$OUT_TRAINED_ORDERED" "./fsrcnn_parallel_ordered_training"

# # 3. Generate Chart
# if [ -d "venv" ]; then
#     echo "========================================================="
#     echo "Generating Stability Chart..."
#     source venv/bin/activate
#     python3 chart.py
#     deactivate
#     echo "Chart generated: psnr_stability_analysis.png"
# else
#     echo "Warning: venv not found. Skipping chart generation."
# fi

# Bersihkan
rm temp_out.yuv
echo "========================================================="
echo "Selesai! Bandingkan nilai 'Min PSNR' di atas."