# Project AION Build System
# Compiler: GCC or Clang
CC = gcc
CFLAGS = -O3 -mcpu=apple-m1 -pthread -Wall -lm -framework Accelerate

# macOS OpenMP support (assuming Homebrew libomp)
# We attempt to link against Homebrew's libomp. 
# If this fails, we fall back to serial execution.
OMP_FLAGS = -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/include -L/opt/homebrew/lib

all: kernel_simd kernel_amx quantize

kernel_simd: src/kernel_simd.c
	@echo "Attempting build with OpenMP..."
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o kernel_simd src/kernel_simd.c 2>/dev/null || (echo "OpenMP build failed. Falling back to serial build..."; $(CC) $(CFLAGS) -o kernel_simd src/kernel_simd.c)

kernel_amx: src/kernel_amx.c
	$(CC) $(CFLAGS) -o kernel_amx src/kernel_amx.c

quantize: quantize.c
	$(CC) $(CFLAGS) -o quantize quantize.c

clean:
	rm -f kernel_simd kernel_amx quantize
