#!/bin/bash
###############################################################################
# comparison352x288.sh - FSRCNN Performance Comparison Script (352x288)
# 
# Menguji executable FSRCNN 288p dengan bobot original dan trained,
# lalu membandingkan: waktu eksekusi, ukuran output, dan PSNR.
#
# Usage: bash comparison352x288.sh
###############################################################################

set -e

# ===================== KONFIGURASI =====================
INPUT_FILE="suzie_cif_288p.yuv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_PATH="${SCRIPT_DIR}/${INPUT_FILE}"

# Resolusi CIF (352x288)
WIDTH=352
HEIGHT=288

# Upscale factor (2x)
OUT_WIDTH=$((WIDTH * 2))
OUT_HEIGHT=$((HEIGHT * 2))

# Jumlah iterasi untuk rata-rata waktu
NUM_RUNS=5

# Daftar executable FSRCNN yang akan diuji
EXECUTABLES=(
    "fsrcnn_parallel_spatial_reduction_288p"
    "fsrcnn_critical_288p"
)

# Label untuk setiap executable (untuk display)
EXE_LABELS=(
    "Parallel Spatial Reduction 288p"
    "OpenMP Critical 288p"
)

# Konfigurasi bobot & bias (original dan trained)
WEIGHT_CONFIGS=("original" "trained")

# File sumber bobot & bias
WEIGHT_ORIGINAL="${SCRIPT_DIR}/weights_layer8_original.txt"
WEIGHT_TRAINED="${SCRIPT_DIR}/weights_layer8_trained.txt"
BIAS_TRAINED="${SCRIPT_DIR}/biasess_layer8_trained.txt"

# File target yang dibaca oleh executable
WEIGHT_TARGET="${SCRIPT_DIR}/weights_layer8.txt"
BIAS_TARGET="${SCRIPT_DIR}/biasess_layer8.txt"

# Backup bias original (karena BIAS_TARGET = biasess_layer8.txt yang sama)
BIAS_ORIGINAL="${SCRIPT_DIR}/biasess_layer8_original.txt"
if [ ! -f "$BIAS_ORIGINAL" ]; then
    cp "${SCRIPT_DIR}/biasess_layer8.txt" "$BIAS_ORIGINAL"
    echo "INFO: Backup bias original ke biasess_layer8_original.txt"
fi

# ===================== VALIDASI =====================
if [ ! -f "$INPUT_PATH" ]; then
    echo "ERROR: Input file '${INPUT_FILE}' tidak ditemukan di ${SCRIPT_DIR}"
    exit 1
fi

for exe in "${EXECUTABLES[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/${exe}" ]; then
        echo "ERROR: Executable '${exe}' tidak ditemukan di ${SCRIPT_DIR}"
        exit 1
    fi
    if [ ! -x "${SCRIPT_DIR}/${exe}" ]; then
        echo "WARNING: '${exe}' tidak executable, menambahkan permission..."
        chmod +x "${SCRIPT_DIR}/${exe}"
    fi
done

# Validasi file bobot & bias
for f in "$WEIGHT_ORIGINAL" "$WEIGHT_TRAINED" "$BIAS_ORIGINAL" "$BIAS_TRAINED"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File bobot/bias '${f}' tidak ditemukan!"
        exit 1
    fi
done

# ===================== FUNGSI UTILITAS =====================

# Fungsi untuk meng-copy bobot & bias sesuai konfigurasi
setup_weights() {
    local config="$1"
    if [ "$config" == "original" ]; then
        cp "$WEIGHT_ORIGINAL" "$WEIGHT_TARGET"
        cp "$BIAS_ORIGINAL" "$BIAS_TARGET"
    elif [ "$config" == "trained" ]; then
        cp "$WEIGHT_TRAINED" "$WEIGHT_TARGET"
        cp "$BIAS_TRAINED" "$BIAS_TARGET"
    fi
}

# Fungsi untuk menghitung PSNR antara dua file YUV
calculate_psnr() {
    local file1="$1"
    local file2="$2"
    
    if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
        echo "N/A"
        return
    fi
    
    # Cek apakah ffmpeg tersedia untuk PSNR
    if command -v ffmpeg &> /dev/null; then
        local psnr_output
        psnr_output=$(ffmpeg -s ${OUT_WIDTH}x${OUT_HEIGHT} -pix_fmt yuv420p -i "$file1" \
                             -s ${OUT_WIDTH}x${OUT_HEIGHT} -pix_fmt yuv420p -i "$file2" \
                             -lavfi psnr -f null - 2>&1 | grep "average" | tail -1)
        
        if [ -n "$psnr_output" ]; then
            echo "$psnr_output" | sed 's/.*average://' | awk '{print $1}'
        else
            echo "N/A"
        fi
    else
        echo "ffmpeg not found"
    fi
}

# Fungsi untuk mendapatkan waktu eksekusi dalam milidetik
get_time_ms() {
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v gdate &> /dev/null; then
            echo $(($(gdate +%s%N) / 1000000))
        elif command -v python3 &> /dev/null; then
            python3 -c "import time; print(int(time.time() * 1000))"
        else
            echo $(($(date +%s) * 1000))
        fi
    else
        echo $(($(date +%s%N) / 1000000))
    fi
}

# ===================== HEADER REPORT =====================
echo "============================================================================="
echo "              FSRCNN PERFORMANCE COMPARISON REPORT (352x288)"
echo "============================================================================="
echo ""
echo "Tanggal        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Input File     : ${INPUT_FILE}"
echo "Input Size     : $(du -h "$INPUT_PATH" | awk '{print $1}')"
echo "Resolusi Input : ${WIDTH}x${HEIGHT} (CIF)"
echo "Resolusi Output: ${OUT_WIDTH}x${OUT_HEIGHT}"
echo "Jumlah Iterasi : ${NUM_RUNS} (untuk rata-rata waktu)"
echo "OS             : $(uname -s) $(uname -m)"
echo "Pengujian      : Setiap executable diuji dengan bobot Original & Trained"
echo ""

# ===================== EKSEKUSI & PENGUKURAN =====================

# Total test cases = executables x weight_configs
TOTAL_TESTS=$(( ${#EXECUTABLES[@]} * ${#WEIGHT_CONFIGS[@]} ))

# Array untuk menyimpan hasil (index = test case index)
declare -a ALL_LABELS
declare -a TIMES_AVG
declare -a TIMES_MIN
declare -a TIMES_MAX
declare -a OUTPUT_SIZES
declare -a OUTPUT_FILES
declare -a PSNR_VALUES

# Buat ground truth (menggunakan executable pertama dengan bobot original)
GROUND_TRUTH="${SCRIPT_DIR}/output_ground_truth_352x288.yuv"
echo ">> Membuat ground truth (${EXE_LABELS[0]} + Original)..."
setup_weights "original"
"${SCRIPT_DIR}/${EXECUTABLES[0]}" "$INPUT_PATH" "$GROUND_TRUTH" > /dev/null 2>&1
echo "   Ground truth: $(du -h "$GROUND_TRUTH" | awk '{print $1}')"
echo ""

echo "============================================================================="
echo "                         MENJALANKAN BENCHMARK"
echo "============================================================================="
echo ""

test_idx=0

for i in "${!EXECUTABLES[@]}"; do
    exe="${EXECUTABLES[$i]}"
    exe_label="${EXE_LABELS[$i]}"
    
    for wconfig in "${WEIGHT_CONFIGS[@]}"; do
        label="${exe_label} [${wconfig}]"
        ALL_LABELS[$test_idx]="$label"
        output_file="${SCRIPT_DIR}/output_${exe}_${wconfig}.yuv"
        OUTPUT_FILES[$test_idx]="$output_file"
        
        echo "---------------------------------------------------------------------"
        echo "[$((test_idx+1))/${TOTAL_TESTS}] Testing: ${label}"
        echo "     Executable: ${exe}"
        echo "     Bobot/Bias: ${wconfig}"
        echo "---------------------------------------------------------------------"
        
        # Copy bobot & bias sesuai konfigurasi
        setup_weights "$wconfig"
        
        total_time=0
        min_time=999999999
        max_time=0
        
        for run in $(seq 1 $NUM_RUNS); do
            # Hapus output sebelumnya
            rm -f "$output_file"
            
            # Ukur waktu eksekusi
            start_time=$(get_time_ms)
            "${SCRIPT_DIR}/${exe}" "$INPUT_PATH" "$output_file" > /dev/null 2>&1
            end_time=$(get_time_ms)
            
            elapsed=$((end_time - start_time))
            total_time=$((total_time + elapsed))
            
            if [ $elapsed -lt $min_time ]; then
                min_time=$elapsed
            fi
            if [ $elapsed -gt $max_time ]; then
                max_time=$elapsed
            fi
            
            printf "     Run %d/%d: %d ms\n" "$run" "$NUM_RUNS" "$elapsed"
        done
        
        avg_time=$((total_time / NUM_RUNS))
        TIMES_AVG[$test_idx]=$avg_time
        TIMES_MIN[$test_idx]=$min_time
        TIMES_MAX[$test_idx]=$max_time
        
        # Ukuran output
        if [ -f "$output_file" ]; then
            OUTPUT_SIZES[$test_idx]=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null || echo "0")
        else
            OUTPUT_SIZES[$test_idx]=0
        fi
        
        echo ""
        printf "     Rata-rata : %d ms\n" "$avg_time"
        printf "     Minimum   : %d ms\n" "$min_time"
        printf "     Maximum   : %d ms\n" "$max_time"
        printf "     Output    : %s bytes\n" "${OUTPUT_SIZES[$test_idx]}"
        echo ""
        
        test_idx=$((test_idx + 1))
    done
done

# ===================== PSNR COMPARISON =====================
echo "============================================================================="
echo "                       MENGHITUNG PSNR"
echo "============================================================================="
echo ""

for idx in $(seq 0 $((TOTAL_TESTS - 1))); do
    label="${ALL_LABELS[$idx]}"
    output_file="${OUTPUT_FILES[$idx]}"
    
    printf "  Menghitung PSNR: %-45s ... " "${label}"
    psnr=$(calculate_psnr "$GROUND_TRUTH" "$output_file")
    PSNR_VALUES[$idx]="$psnr"
    echo "$psnr dB"
done
echo ""

# ===================== TABEL RINGKASAN =====================
echo "============================================================================="
echo "                         RINGKASAN HASIL"
echo "============================================================================="
echo ""

# Header tabel
printf "%-45s | %10s | %10s | %10s | %12s | %10s\n" \
    "Metode" "Avg (ms)" "Min (ms)" "Max (ms)" "Output (B)" "PSNR (dB)"
printf "%-45s-+-%10s-+-%10s-+-%10s-+-%12s-+-%10s\n" \
    "---------------------------------------------" "----------" "----------" "----------" "------------" "----------"

for idx in $(seq 0 $((TOTAL_TESTS - 1))); do
    label="${ALL_LABELS[$idx]}"
    printf "%-45s | %10d | %10d | %10d | %12s | %10s\n" \
        "${label}" \
        "${TIMES_AVG[$idx]}" \
        "${TIMES_MIN[$idx]}" \
        "${TIMES_MAX[$idx]}" \
        "${OUTPUT_SIZES[$idx]}" \
        "${PSNR_VALUES[$idx]}"
done

echo ""

# ===================== SPEEDUP ANALYSIS =====================
echo "============================================================================="
echo "                       ANALISIS SPEEDUP"
echo "============================================================================="
echo ""

# Gunakan test case pertama sebagai baseline
baseline=${TIMES_AVG[0]}

printf "Baseline: %s (%d ms)\n\n" "${ALL_LABELS[0]}" "$baseline"
printf "%-45s | %12s | %s\n" "Metode" "Speedup" "Keterangan"
printf "%-45s-+-%12s-+-%s\n" "---------------------------------------------" "------------" "--------------------"

for idx in $(seq 0 $((TOTAL_TESTS - 1))); do
    label="${ALL_LABELS[$idx]}"
    avg=${TIMES_AVG[$idx]}
    
    if [ "$avg" -gt 0 ]; then
        speedup=$(awk "BEGIN {printf \"%.2f\", $baseline / $avg}")
        
        if (( $(awk "BEGIN {print ($speedup >= 1.0) ? 1 : 0}") )); then
            keterangan="${speedup}x lebih cepat"
        else
            slowdown=$(awk "BEGIN {printf \"%.2f\", $avg / $baseline}")
            keterangan="${slowdown}x lebih lambat"
        fi
    else
        speedup="N/A"
        keterangan="Error"
    fi
    
    printf "%-45s | %12s | %s\n" "${label}" "${speedup}x" "$keterangan"
done

echo ""

# ===================== OUTPUT CONSISTENCY CHECK =====================
echo "============================================================================="
echo "                    CEK KONSISTENSI OUTPUT"
echo "============================================================================="
echo ""

echo "Membandingkan output setiap metode dengan ground truth..."
echo ""

for idx in $(seq 0 $((TOTAL_TESTS - 1))); do
    label="${ALL_LABELS[$idx]}"
    output_file="${OUTPUT_FILES[$idx]}"
    
    if [ -f "$output_file" ] && [ -f "$GROUND_TRUTH" ]; then
        if cmp -s "$GROUND_TRUTH" "$output_file"; then
            printf "  %-45s : IDENTIK ✓\n" "${label}"
        else
            diff_bytes=$(cmp -l "$GROUND_TRUTH" "$output_file" 2>/dev/null | wc -l | tr -d ' ')
            printf "  %-45s : BERBEDA ✗ (%s bytes berbeda)\n" "${label}" "$diff_bytes"
        fi
    else
        printf "  %-45s : FILE TIDAK DITEMUKAN\n" "${label}"
    fi
done

echo ""
echo "============================================================================="
echo "                       BENCHMARK SELESAI"
echo "============================================================================="
echo ""
echo "Output files tersimpan di: ${SCRIPT_DIR}/"
for idx in $(seq 0 $((TOTAL_TESTS - 1))); do
    echo "  - $(basename "${OUTPUT_FILES[$idx]}")"
done
echo "  - $(basename "$GROUND_TRUTH") (ground truth)"
echo ""
