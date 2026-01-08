import numpy as np
import os
import sys
import struct

def quantize_weights(weights, group_size=32):
    """
    Performs Group-wise Symmetric Quantization (G=32).
    
    Theoretical Basis:
    We map the continuous FP32 range of a weight block to the discrete integer set {-7, ..., +7}.
    This preserves the distribution's shape while reducing bit-width.
    
    Scaling Dynamics:
    We use per-channel (or per-group) scaling factors to mitigate the accuracy loss 
    caused by outliers. By partitioning the tensor into small groups (G=32), 
    outliers only distort the quantization grid of their local neighborhood.
    
    Returns: (packed_weights, scales)
    """
    # Flatten just in case, though we assume flat arrays
    w_flat = weights.flatten()
    
    # Pad to group_size
    padding = (group_size - (len(w_flat) % group_size)) % group_size
    if padding > 0:
        w_flat = np.concatenate([w_flat, np.zeros(padding, dtype=np.float32)])
    
    # Reshape into groups
    n_groups = len(w_flat) // group_size
    w_groups = w_flat.reshape(n_groups, group_size)
    
    # Calculate scales: max(abs(w_group)) / 7.0
    # Int4 range is [-7, 7] (technically -8 to 7, but we use symmetric so -7 to 7)
    scales = np.max(np.abs(w_groups), axis=1) / 7.0
    scales[scales == 0] = 1.0 # Avoid div by zero
    
    # Quantize: w / scale
    # shape: (n_groups, group_size) / (n_groups, 1)
    w_q = w_groups / scales[:, np.newaxis]
    w_q = np.round(w_q).astype(np.int8)
    w_q = np.clip(w_q, -7, 7)
    
    # Pack: 2 int4 per uint8
    # We take valid range 0-15 by adding 8? No, let's keep it signed for now.
    # 2's complement 4-bit is tricky to pack simply.
    # Alternative: Offset binary. Add 8 to get 0..15 range.
    # -7 -> 1, 0 -> 8, 7 -> 15. -8 -> 0 is unused.
    w_q_offset = w_q + 8 
    
    # Packing Logic:
    # To store 4-bit values in byte-addressable memory, we must pack two weights per byte.
    # Architecture Decision: Little-Endian Nibble Ordering
    # - Low Nibble (Bits 0-3): Corresponds to w[2i]
    # - High Nibble (Bits 4-7): Corresponds to w[2i+1]
    # This aligns with the "vld1q_u8" instruction's behavior during unpacking in the C kernel.
    w_q_offset_flat = w_q_offset.flatten().astype(np.uint8)
    
    # Separate evens and odds
    low_nibbles = w_q_offset_flat[0::2]
    high_nibbles = w_q_offset_flat[1::2]
    
    packed = (low_nibbles & 0xF) | ((high_nibbles & 0xF) << 4)
    
    return packed, scales.astype(np.float32)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_model.bin> <output_model_int4.bin>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Read Config
    # Model format:
    # Config struct (7 ints = 28 bytes)
    # Weights (float32)
    
    config_dtype = np.dtype([
        ('dim', 'i4'),
        ('hidden_dim', 'i4'),
        ('n_layers', 'i4'),
        ('n_heads', 'i4'),
        ('n_kv_heads', 'i4'),
        ('vocab_size', 'i4'),
        ('seq_len', 'i4')
    ])
    
    with open(input_path, 'rb') as f:
        config_data = f.read(28)
        config = np.frombuffer(config_data, dtype=config_dtype)[0]
        
    print(f"Loaded config: {config}")
    
    # Read all weights
    # We need to read them in order and quantize specific ones
    # Order from C code:
    # token_embedding_table: vocab * dim
    # rms_att_weight: layers * dim
    # wq: layers * dim * n_heads * head_size
    # wk: ...
    # wv: ...
    # wo: ...
    # rms_ffn_weight: layers * dim
    # w1: ...
    # w2: ...
    # w3: ...
    # rms_final_weight: dim
    # freq_cis_real: seq_len * head_size / 2
    # freq_cis_imag: seq_len * head_size / 2
    
    # Weights to Quantize:
    # wq, wk, wv, wo, w1, w2, w3, token_embedding_table?
    # Usually we don't quantize norms or embedding output unless necessary.
    # For simplicity and "8x bandwidth", let's quantize the big matrices:
    # wq, wk, wv, wo, w1, w2, w3.
    # The others are small (vectors or smaller matrices).
    # Token embedding is big (vocab * dim). Let's quantize it too if we want full impact.
    
    # Re-reading file using numpy memmap or just read
    full_data = np.fromfile(input_path, dtype=np.float32, offset=28)
    
    # Pointers
    offset = 0
    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    vocab_size = abs(config['vocab_size'])
    seq_len = config['seq_len']
    head_size = dim // n_heads
    
    # Helper to slice
    def get_slice(size):
        nonlocal offset
        s = full_data[offset : offset + size]
        offset += size
        return s

    with open(output_path, 'wb') as fout:
        # Write Config
        fout.write(config_data)
        
        # 1. token_embedding_table (vocab * dim) -> Quantize? 
        # Yes, big.
        w = get_slice(vocab_size * dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes()) # Write scales first
        fout.write(p.tobytes()) # Then packed weights
        print(f"Quantized token_embedding_table: {w.nbytes} -> {p.nbytes + s.nbytes}")

        # 2. rms_att_weight (layers * dim) -> KEEP F32 (small)
        w = get_slice(n_layers * dim)
        fout.write(w.tobytes())
        
        # 3. wq (layers * dim * dim) -> Quantize
        w = get_slice(n_layers * dim * dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 4. wk (layers * dim * kv_dim) -> Quantize
        kv_dim = (dim * n_kv_heads) // n_heads
        w = get_slice(n_layers * dim * kv_dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 5. wv (layers * dim * kv_dim) -> Quantize
        w = get_slice(n_layers * dim * kv_dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 6. wo (layers * dim * dim) -> Quantize
        w = get_slice(n_layers * dim * dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 7. rms_ffn_weight (layers * dim) -> KEEP F32
        w = get_slice(n_layers * dim)
        fout.write(w.tobytes())
        
        # 8. w1 (layers * dim * hidden_dim) -> Quantize
        w = get_slice(n_layers * dim * hidden_dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 9. w2 (layers * hidden_dim * dim) -> Quantize
        w = get_slice(n_layers * hidden_dim * dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 10. w3 (layers * dim * hidden_dim) -> Quantize
        w = get_slice(n_layers * dim * hidden_dim)
        p, s = quantize_weights(w)
        fout.write(s.tobytes())
        fout.write(p.tobytes())
        
        # 11. rms_final_weight (dim) -> KEEP F32
        w = get_slice(dim)
        fout.write(w.tobytes())
        
        # 12. freq_cis_real
        w = get_slice(seq_len * (head_size // 2))
        fout.write(w.tobytes())
        
        # 13. freq_cis_imag
        w = get_slice(seq_len * (head_size // 2))
        fout.write(w.tobytes())

    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
