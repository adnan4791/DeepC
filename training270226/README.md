# 1. Build Layer 7 and Layer 8
#gcc-15 -shared -o fsrcnn_layer8.so source_layer8_so.c -O3 -fPIC -fopenmp
gcc-15 -shared -o fsrcnn_race_sim.so -fPIC fsrcnn_race_sim.c -fopenmp -O2
gcc-15 -o dump_layer7 source_dump_layer7.c -fopenmp -lm

# 2. Generate training data
./dump_layer7 suzie_qcif.yuv ground_truth/suzie_qcif_serial_2.yuv 150

# 3. Run training
python3 train_layer8_race_condition.py

# 4. Compile the source C FSRCNN
gcc-15 fsrcnn_parallel.c -o fsrcnn_parallel_150 -fopenmp -lm -O3

# 5. Build ground truth
## Read the README.md in ground_truth folder for more information
gcc-15 ./ground_truth/fsrcnn_serial_layer_8.c -o ./ground_truth/fsrcnn_serial_layer_8_150 -fopenmp -lm -O3
./ground_truth/fsrcnn_serial_layer_8_150 suzie_qcif.yuv ./ground_truth/suzie_qcif_serial.yuv

# 6. Verify race condition
./verify_race.sh
