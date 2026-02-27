#!/bin/bash
###############################################################################
# comparation.sh - FSRCNN Performance Comparison Script
# 
# Menguji seluruh executable FSRCNN di folder ini dengan input yang sama
# dan membandingkan: waktu eksekusi, ukuran output, dan PSNR.
#
# Usage: bash comparation.sh
###############################################################################

set -e

# ===================== KONFIGURASI =====================
INPUT_FILE="suzie_qcif.yuv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_PATH="${SCRIPT_DIR}/${INPUT_FILE}"

# Resolusi QCIF
WIDTH=176
HEIGHT=144

# Upscale factor (2x)
OUT_WIDTH=$((WIDTH * 2))
OUT_HEIGHT=$((HEIGHT * 2))

# Jumlah iterasi untuk rata-rata waktu
NUM_RUNS=5

# Daftar executable FSRCNN yang akan diuji
EXECUTABLES=(
    "fsrcnn_serial_layer_8_150"
    "fsrcnn_parallel_spatial_reduction_150"
    "fsrcnn_parallel_spatial_reduction_1loop_150"
    "fsrcnn_parallel_spatial_reduction_1loop_150_2"
    "fsrcnn_critical_150"
    "fsrcnn_taskloop_150"
)

# Label untuk setiap executable (untuk display)
LABELS=(
    "Serial (Layer 8)"
    "Parallel Spatial Reduction"
    "Parallel Spatial Reduction 1 Loop"
    "Parallel Spatial Reduction 1 Loop 2"
    "OpenMP Critical"
    "OpenMP Taskloop"
)

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

# ===================== FUNGSI UTILITAS =====================

# Fungsi untuk menghitung PSNR antara dua file YUV
# Menggunakan perbandingan byte-level sederhana
calculate_psnr() {
    local file1="$1"
    local file2="$2"
    
    if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
        echo "N/A"
        return
    fi
    
    # Cek apakah ffmpeg tersedia untuk PSNR
    if command -v ffmpeg &> /dev/null; then
        # Gunakan ffmpeg untuk menghitung PSNR
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
        # macOS: gunakan gdate jika tersedia, atau python
        if command -v gdate &> /dev/null; then
            echo $(($(gdate +%s%N) / 1000000))
        elif command -v python3 &> /dev/null; then
            python3 -c "import time; print(int(time.time() * 1000))"
        else
            # Fallback: gunakan detik saja
            echo $(($(date +%s) * 1000))
        fi
    else
        echo $(($(date +%s%N) / 1000000))
    fi
}

# ===================== HEADER REPORT =====================
echo "============================================================================="
echo "              FSRCNN PERFORMANCE COMPARISON REPORT"
echo "============================================================================="
echo ""
echo "Tanggal        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Input File     : ${INPUT_FILE}"
echo "Input Size     : $(du -h "$INPUT_PATH" | awk '{print $1}')"
echo "Resolusi Input : ${WIDTH}x${HEIGHT} (QCIF)"
echo "Resolusi Output: ${OUT_WIDTH}x${OUT_HEIGHT}"
echo "Jumlah Iterasi : ${NUM_RUNS} (untuk rata-rata waktu)"
echo "OS             : $(uname -s) $(uname -m)"
echo ""

# ===================== EKSEKUSI & PENGUKURAN =====================

# Array untuk menyimpan hasil
declare -a TIMES_AVG
declare -a TIMES_MIN
declare -a TIMES_MAX
declare -a OUTPUT_SIZES
declare -a OUTPUT_FILES

# Buat ground truth dari serial (eksekusi pertama serial sebagai referensi)
GROUND_TRUTH="${SCRIPT_DIR}/output_serial_ground_truth.yuv"
echo ">> Membuat ground truth dari Serial..."
"${SCRIPT_DIR}/fsrcnn_serial_layer_8_150" "$INPUT_PATH" "$GROUND_TRUTH" > /dev/null 2>&1
echo "   Ground truth: $(du -h "$GROUND_TRUTH" | awk '{print $1}')"
echo ""

echo "============================================================================="
echo "                         MENJALANKAN BENCHMARK"
echo "============================================================================="
echo ""

for i in "${!EXECUTABLES[@]}"; do
    exe="${EXECUTABLES[$i]}"
    label="${LABELS[$i]}"
    output_file="${SCRIPT_DIR}/output_${exe}.yuv"
    OUTPUT_FILES[$i]="$output_file"
    
    echo "---------------------------------------------------------------------"
    echo "[$((i+1))/${#EXECUTABLES[@]}] Testing: ${label}"
    echo "     Executable: ${exe}"
    echo "---------------------------------------------------------------------"
    
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
    TIMES_AVG[$i]=$avg_time
    TIMES_MIN[$i]=$min_time
    TIMES_MAX[$i]=$max_time
    
    # Ukuran output
    if [ -f "$output_file" ]; then
        OUTPUT_SIZES[$i]=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null || echo "0")
    else
        OUTPUT_SIZES[$i]=0
    fi
    
    echo ""
    printf "     Rata-rata : %d ms\n" "$avg_time"
    printf "     Minimum   : %d ms\n" "$min_time"
    printf "     Maximum   : %d ms\n" "$max_time"
    printf "     Output    : %s bytes\n" "${OUTPUT_SIZES[$i]}"
    echo ""
done

# ===================== PSNR COMPARISON =====================
echo "============================================================================="
echo "                       MENGHITUNG PSNR"
echo "============================================================================="
echo ""

declare -a PSNR_VALUES

for i in "${!EXECUTABLES[@]}"; do
    label="${LABELS[$i]}"
    output_file="${OUTPUT_FILES[$i]}"
    
    printf "  Menghitung PSNR: %-35s ... " "${label}"
    psnr=$(calculate_psnr "$GROUND_TRUTH" "$output_file")
    PSNR_VALUES[$i]="$psnr"
    echo "$psnr dB"
done
echo ""

# ===================== TABEL RINGKASAN =====================
echo "============================================================================="
echo "                         RINGKASAN HASIL"
echo "============================================================================="
echo ""

# Header tabel
printf "%-35s | %10s | %10s | %10s | %12s | %10s\n" \
    "Metode" "Avg (ms)" "Min (ms)" "Max (ms)" "Output (B)" "PSNR (dB)"
printf "%-35s-+-%10s-+-%10s-+-%10s-+-%12s-+-%10s\n" \
    "-----------------------------------" "----------" "----------" "----------" "------------" "----------"

# Cari waktu tercepat untuk speedup
fastest=${TIMES_AVG[0]}
for t in "${TIMES_AVG[@]}"; do
    if [ "$t" -lt "$fastest" ]; then
        fastest=$t
    fi
done

for i in "${!EXECUTABLES[@]}"; do
    label="${LABELS[$i]}"
    printf "%-35s | %10d | %10d | %10d | %12s | %10s\n" \
        "${label}" \
        "${TIMES_AVG[$i]}" \
        "${TIMES_MIN[$i]}" \
        "${TIMES_MAX[$i]}" \
        "${OUTPUT_SIZES[$i]}" \
        "${PSNR_VALUES[$i]}"
done

echo ""

# ===================== SPEEDUP ANALYSIS =====================
echo "============================================================================="
echo "                       ANALISIS SPEEDUP"
echo "============================================================================="
echo ""

# Gunakan serial sebagai baseline
baseline=${TIMES_AVG[0]}

printf "Baseline: %s (%d ms)\n\n" "${LABELS[0]}" "$baseline"
printf "%-35s | %12s | %s\n" "Metode" "Speedup" "Keterangan"
printf "%-35s-+-%12s-+-%s\n" "-----------------------------------" "------------" "--------------------"

for i in "${!EXECUTABLES[@]}"; do
    label="${LABELS[$i]}"
    avg=${TIMES_AVG[$i]}
    
    if [ "$avg" -gt 0 ]; then
        # Hitung speedup dengan 2 desimal menggunakan awk
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
    
    printf "%-35s | %12s | %s\n" "${label}" "${speedup}x" "$keterangan"
done

echo ""

# ===================== OUTPUT CONSISTENCY CHECK =====================
echo "============================================================================="
echo "                    CEK KONSISTENSI OUTPUT"
echo "============================================================================="
echo ""

echo "Membandingkan output setiap metode dengan ground truth (Serial)..."
echo ""

for i in "${!EXECUTABLES[@]}"; do
    label="${LABELS[$i]}"
    output_file="${OUTPUT_FILES[$i]}"
    
    if [ -f "$output_file" ] && [ -f "$GROUND_TRUTH" ]; then
        if cmp -s "$GROUND_TRUTH" "$output_file"; then
            printf "  %-35s : IDENTIK ✓\n" "${label}"
        else
            diff_bytes=$(cmp -l "$GROUND_TRUTH" "$output_file" 2>/dev/null | wc -l | tr -d ' ')
            printf "  %-35s : BERBEDA ✗ (%s bytes berbeda)\n" "${label}" "$diff_bytes"
        fi
    else
        printf "  %-35s : FILE TIDAK DITEMUKAN\n" "${label}"
    fi
done

echo ""
echo "============================================================================="
echo "                       BENCHMARK SELESAI"
echo "============================================================================="
echo ""
echo "Output files tersimpan di: ${SCRIPT_DIR}/"
for i in "${!EXECUTABLES[@]}"; do
    echo "  - output_${EXECUTABLES[$i]}.yuv"
done
echo "  - output_serial_ground_truth.yuv (ground truth)"
echo ""
