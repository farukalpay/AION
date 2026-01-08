
import sys
import struct
import numpy as np
import time

def qweight_dequantize(scales, packed, dim):
    """
    Naive Reference Implementation of Dequantization.
    
    Pedagogical Note:
    This function demonstrates the "Interpretation Overhead" of high-level languages.
    Performing bitwise operations (masking/shifting) on millions of elements 
    in an interpreted loop (or even via NumPy broadcasting) incurs significant 
    syscall/allocation latency compared to the single-cycle NEON instructions in C.
    """
    # scales: [dim/32]
    # packed: [dim/2]
    # output: [dim]
    
    # Vectorization Strategy:
    # 1. Unpack uint8 -> 2x int8 using bitwise ops (High/Low nibbles).
    # 2. Interleave: Combine even/odd indices to reconstruct the flat weight vector.
    # 3. Scale: Broadcast the group scale factor to all 32 elements in the group.
    
    # Unpack uint8 to 2x int8
    packed_flat = np.frombuffer(packed, dtype=np.uint8)
    low = (packed_flat & 0xF).astype(np.int8) - 8
    high = (packed_flat >> 4).astype(np.int8) - 8
    
    # Interleave low and high (Columnar Reassembly)
    weights_int = np.stack((low, high), axis=1).flatten()
    
    # Broadcast scales (Stride-32 repetition)
    scales_expanded = np.repeat(scales, 32)
    
    return weights_int * scales_expanded

class Int4Linear:
    def __init__(self, scales, packed, dim_in, dim_out):
        self.scales = np.frombuffer(scales, dtype=np.float32)
        self.packed = packed
        self.dim_in = dim_in # n
        self.dim_out = dim_out # d
        
        # Checking shapes
        # Packed should be (dim_out, dim_in/2)?
        # In our file format, weights are flattened.
        # But QWeight struct was:
        # float* scales;   // [dim / 32]
        # uint8_t* packed; // [dim / 2]
        # This struct represents a SINGLE ROW? No, it represents the whole tensor in C?
        # Wait, in C `QWeight` was defined as:
        # typedef struct { float* scales; uint8_t* packed; } QWeight;
        # And mapped via:
        # MAP_Q(weights->wq, config->n_layers * config->dim * config->dim);
        # So `wq` is one giant QWeight struct for the whole layer stack?
        # Yes.
        pass

    def dequantize(self):
        # Full dequantization (Memory intensive!)
        # In a real generic python runner, we might dequantize per matmul
        pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_python_int4.py <model_path> [steps]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    steps = 10
    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])
    
    # Methodology:
    # We simulate a "Memory-Constrained" environment where weights MUST remain 
    # compressed in RAM and are only decompressed Just-In-Time (JIT) for computation.
    # This exposes the CPU overhead of decompression, isolating the "Compute Wall".
    
    
    print("Benchmarking Python Int4 (Emulated Dequantization)...")
    
    # Dummy workload to measure "Overhead of Dequantization"
    # MatMul: [1, 288] x [288, 288] (Tiny)
    # Let's align with C bench: [1, 2048] x [2048, 2048]
    dim = 2048
    
    # Generate random Int4 weights
    # 2048*2048 weights
    n_elements = dim * dim
    packed_size = n_elements // 2
    scales_size = n_elements // 32
    
    packed = np.random.randint(0, 255, size=packed_size, dtype=np.uint8).tobytes()
    scales = np.random.rand(scales_size).astype(np.float32).tobytes()
    
    x = np.random.rand(dim).astype(np.float32)
    
    t0 = time.time()
    
    for i in range(steps):
        # 1. Dequantize (Simulating "Just-in-Time" decompression)
        w_scales = np.frombuffer(scales, dtype=np.float32)
        
        # Numpy overhead is huge here, but it's "Python".
        packed_np = np.frombuffer(packed, dtype=np.uint8)
        low = (packed_np & 0xF).astype(np.int8) - 8
        high = (packed_np >> 4).astype(np.int8) - 8
        w_int = np.stack((low, high), axis=1).flatten()
        
        w_float = w_int * np.repeat(w_scales, 32)
        w_matrix = w_float.reshape(dim, dim)
        
        # 2. Matmul
        res = np.dot(x, w_matrix.T)
        
    t1 = time.time()
    print(f"Python Int4 (JIT Dequant): {steps / (t1-t0):.4f} iter/s")
    
    # Log Int4
    with open("benchmarks.csv", "a") as f:
        import datetime
        ts = datetime.datetime.now().ctime()
        f.write(f"{ts},Py_Int4_Emulated,1,{steps},{t1-t0:.4f},{steps/(t1-t0):.2f}\n")

    # Comparison: Pre-dequantized (F32)
    w_f32 = np.random.rand(dim, dim).astype(np.float32)
    t0 = time.time()
    for i in range(steps):
        res = np.dot(x, w_f32.T)
    t1 = time.time()
    print(f"Python F32 (Baseline): {steps / (t1-t0):.4f} iter/s")
    
    # Log F32
    with open("benchmarks.csv", "a") as f:
        import datetime
        ts = datetime.datetime.now().ctime()
        f.write(f"{ts},Py_F32_BLAS,1,{steps},{t1-t0:.4f},{steps/(t1-t0):.2f}\n")

    # W4A8 Simulation
    # W4A8 requires quantizing X to Int8 and W to Int8, then using integer dot.
    # We can simulate this by casting to int8 and doing dot.
    print("Benchmarking W4A8 (Emulated)...")
    q_x = (x * 127).astype(np.int8)
    q_w = (np.random.rand(dim, dim) * 127).astype(np.int8)
    
    t0 = time.time()
    for i in range(steps):
        # We assume X needs to be re-quantized each step in a real scenario,
        # but here X is constant. Let's include quantization cost `q_x = ...` inside loop?
        # Yes, to be fair.
        qx_curr = (x * 127).astype(np.int8)
        # Integer matmul
        res = np.dot(qx_curr, q_w.T) 
        # Result is int32. Scale back.
        res_f = res * (1.0/127.0) * (1.0/127.0)
    t1 = time.time()
    print(f"Python W4A8 (Int8 Dot): {steps / (t1-t0):.4f} iter/s")
    
    with open("benchmarks.csv", "a") as f:
        import datetime
        ts = datetime.datetime.now().ctime()
        f.write(f"{ts},Py_W4A8_Emulated,1,{steps},{t1-t0:.4f},{steps/(t1-t0):.2f}\n")

if __name__ == "__main__":
    main()
