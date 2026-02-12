# How to run DeepC with Docker

Since OpenMP (`omp.h`) is not supported by default on macOS clang, we use a GCC Docker container.

## Prerequisites
- Docker Installed
- Docker Compose Installed

## Steps

1. **Start the Container**
   ```bash
   docker-compose up -d --build
   ```

2. **Compile the Code**
   Run the following command to compile `source.c` using the Makefile:
   ```bash
   docker-compose exec deepc make
   ```

3. **Run the Program**
   Run the compiled executable. Replace `input.yuv` and `output.yuv` with your actual file names.
   ```bash
   docker-compose exec deepc ./active_deepc akiyo_cif.yuv output.yuv
   ```

4. **Stop the Container**
   When finished:
   ```bash
   docker-compose down
   ```

## Troubleshooting
- Ensure all `weights_layer*.txt` and `biasess_layer*.txt` files are in the same directory.
- The output file will be created in the local directory since it is mounted to `/app`.
