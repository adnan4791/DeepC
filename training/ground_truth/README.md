# 1. Compile the source C SERIAL FSRCNN
gcc-15 ground_truth/fsrcnn_serial_layer_8.c -o ground_truth/fsrcnn_serial_layer_8 -fopenmp -lm

# 2. Build ground truth
./ground_truth/fsrcnn_serial_layer_8 tulips_yuv420_prog_planar_qcif.yuv ./ground_truth/ground_truth.yuv