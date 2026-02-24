# Makefile for FSRCNN Layer 8 Training
# Compiles the C code for training with race condition compensation

CC = gcc-15
CFLAGS = -O3 -fPIC -fopenmp
LDFLAGS = -fopenmp

# Targets
all: fsrcnn_layer8.so dump_layer7

# Shared library for Python training
# gcc-15 -shared -o fsrcnn_layer8.so source_layer8_so.c -O3 -fPIC -fopenmp
fsrcnn_layer8.so: source_layer8_so.c
	$(CC) -shared -o $@ $< $(CFLAGS)

# Executable to dump Layer 7 outputs
# gcc-15 -o dump_layer7 source_dump_layer7.c -fopenmp -lm
dump_layer7: source_dump_layer7.c
	$(CC) -o $@ $< $(LDFLAGS) -lm

# Original FSRCNN (for comparison)
fsrcnn_original: source_4.c
	$(CC) -o $@ $< $(LDFLAGS) -lm

# Clean
clean:
	rm -f fsrcnn_layer8.so dump_layer7 fsrcnn_original
	rm -rf training_data

# Create training data
training_data: dump_layer7
	./dump_layer7 tulips_yuv420_prog_planar_qcif.yuv output_training.yuv 6

# Run training
train: training_data
	python3 train_layer8_race_condition.py

# Help
help:
	@echo "FSRCNN Layer 8 Training - Race Condition Compensation"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all targets"
	@echo "  fsrcnn_layer8.so - Build shared library for Python"
	@echo "  dump_layer7    - Build executable to dump Layer 7 outputs"
	@echo "  training_data  - Generate training data from video"
	@echo "  train          - Run the training script"
	@echo "  clean          - Remove built files"
	@echo ""
	@echo "Usage:"
	@echo "  1. make all"
	@echo "  2. make training_data  (or manually: ./dump_layer7 input.yuv output.yuv num_frames)"
	@echo "  3. make train          (or manually: python3 train_layer8_race_condition.py)"
	@echo ""
	@echo "The trained weights will be saved to weights_layer8_trained.txt"
	@echo "Replace weights_layer8.txt with the trained weights for inference."

.PHONY: all clean training_data train help
