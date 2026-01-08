"""
Project AION: Reference Oracle Implementation

This script serves as the "Ground Truth" for numerical correctness checking. 
It utilizes the High-Level NumPy API, which delegates linear algebra operations 
to the Apple Accelerate Framework (BLAS/LAPACK) on macOS.

Role in Research:
1. Verification: Provides the exact F32 logits to validate the C11 kernel outputs.
2. Baseline: Establishes the upper-bound performance of un-optimized managed code.
   (Note: While Python is slow, NumPy is fast because it is C/Fortran under the hood).
"""

import sys
import struct
import numpy as np

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(seq_len)
    freqs = np.outer(t, freqs)
    freqs_cis = np.exp(1j * freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    RoPE Rotation Strategy: Complex Multiplication
    
    Mathematical Equivalence:
    The Rotary Positional Embedding (RoPE) corresponds to rotating the query/key
    vectors in a complex 2D plane. 
    
    Optimized Vectorization:
    In Python/NumPy, we model this as Element-wise Complex Multiplication:
        (x + iy) * e^{i\theta}
    
    In the C11 Kernel, this translates to a 2x2 Rotation Matrix application:
        [ cos -sin ] [x]
        [ sin  cos ] [y]
    """
    # xq.shape = [dim] -> complex -> [dim/2]
    xq_ = xq.astype(np.float32).reshape(-1, 2)
    xq_ = xq_[:, 0] + 1j * xq_[:, 1]
    
    xk_ = xk.astype(np.float32).reshape(-1, 2)
    xk_ = xk_[:, 0] + 1j * xk_[:, 1]
    
    # Broadcast freqs_cis
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    
    # Back to real
    xq_out = np.stack([xq_out.real, xq_out.imag], axis=-1).flatten()
    xk_out = np.stack([xk_out.real, xk_out.imag], axis=-1).flatten()
    return xq_out, xk_out

class TransformerConfig:
    def __init__(self, f):
        self.dim, self.hidden_dim, self.n_layers, self.n_heads, \
        self.n_kv_heads, self.vocab_size, self.seq_len = \
        struct.unpack('iiiiiii', f.read(28))
        self.vocab_size = abs(self.vocab_size)

class Transformer:
    def __init__(self, checkpoint_path):
        self.load_model(checkpoint_path)
        
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.config = TransformerConfig(f)
            
            # Helper to read weights
            def read_tensor(shape):
                numel = np.prod(shape)
                data = f.read(numel * 4) # float32
                floats = np.frombuffer(data, dtype=np.float32)
                return floats.reshape(shape)
            
            c = self.config
            head_size = c.dim // c.n_heads
            
            self.token_embedding_table = read_tensor((c.vocab_size, c.dim))
            self.rms_att_weight = read_tensor((c.n_layers, c.dim))
            
            self.wq = read_tensor((c.n_layers, c.n_heads * head_size, c.dim))
            self.wk = read_tensor((c.n_layers, c.n_kv_heads * head_size, c.dim))
            self.wv = read_tensor((c.n_layers, c.n_kv_heads * head_size, c.dim))
            self.wo = read_tensor((c.n_layers, c.dim, c.n_heads * head_size))
            
            self.rms_ffn_weight = read_tensor((c.n_layers, c.dim))
            
            self.w1 = read_tensor((c.n_layers, c.hidden_dim, c.dim))
            self.w2 = read_tensor((c.n_layers, c.dim, c.hidden_dim))
            self.w3 = read_tensor((c.n_layers, c.hidden_dim, c.dim))
            
            self.rms_final_weight = read_tensor((c.dim,))
            
            # freq_cis are usually computed, not stored? 
            # In llama2.c they are read from file because they are exported.
            # We must read them to match file pointer offset.
            self.freq_cis_real = read_tensor((c.seq_len, head_size // 2))
            self.freq_cis_imag = read_tensor((c.seq_len, head_size // 2))
            
            # Reconstruct complex freqs for easier usage
            self.freqs_cis = self.freq_cis_real + 1j * self.freq_cis_imag

    def rmsnorm(self, x, weight):
        # x: [dim]
        # weight: [dim]
        ss = np.mean(x**2)
        return x * (1.0 / np.sqrt(ss + 1e-5)) * weight

    def softmax(self, x):
        # numerical stability
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def forward(self, token, pos, state):
        c = self.config
        x = self.token_embedding_table[token].copy()
        
        dim = c.dim
        head_size = dim // c.n_heads
        
        # Get freqs for this position
        freqs_cis = self.freqs_cis[pos] # [head_size/2]
        
        for l in range(c.n_layers):
            
            # Attention RMSNorm
            xb = self.rmsnorm(x, self.rms_att_weight[l])
            
            # QKV
            # wq: [n_heads * head_size, dim]
            # xb: [dim]
            # q: [n_heads * head_size]
            q = np.dot(self.wq[l], xb)
            k = np.dot(self.wk[l], xb)
            v = np.dot(self.wv[l], xb)
            
            # Apply RoPE
            # We need to reshape q, k to [n_heads, head_size] and [n_kv_heads, head_size]
            q = q.reshape(c.n_heads, head_size)
            k = k.reshape(c.n_kv_heads, head_size)
            
            # Apply RoPE to each head
            for h in range(c.n_heads):
               q_real, k_dummy = apply_rotary_emb(q[h], q[h], freqs_cis) # hacky reuse
               q[h] = q_real
               
            for h in range(c.n_kv_heads):
               k_dummy, k_real = apply_rotary_emb(k[h], k[h], freqs_cis)
               k[h] = k_real
            
            # Update KV Cache
            # state['key_cache'][l, pos] = k
            # state['value_cache'][l, pos] = v
            # To simplify, we keep list
            state['key_cache'][l][pos] = k
            
            # v needs reshape
            v = v.reshape(c.n_kv_heads, head_size)
            state['value_cache'][l][pos] = v
            
            # Multi-head Attention
            # Output container
            xb_att = np.zeros(dim)
            
            for h in range(c.n_heads):
                # Get query
                q_h = q[h] # [head_size]
                
                # Calculate scores
                # We need all keys for this head up to pos
                # key_cache[l][:pos+1, h_kv, :]
                kv_head = h // (c.n_heads // c.n_kv_heads)
                
                keys = state['key_cache'][l][:pos+1, kv_head, :] # [seq, head_size]
                
                # scores = keys @ q_h
                scores = np.dot(keys, q_h) / np.sqrt(head_size)
                
                # Softmax
                scores = self.softmax(scores)
                
                # Weighted sum
                values = state['value_cache'][l][:pos+1, kv_head, :]
                out_h = np.dot(values.T, scores) # [head_size]
                
                xb_att[h*head_size : (h+1)*head_size] = out_h
            
            # Output projection
            xb2 = np.dot(self.wo[l], xb_att)
            
            # Residual 1
            x = x + xb2
            
            # FFN RMSNorm
            xb = self.rmsnorm(x, self.rms_ffn_weight[l])
            
            # FFN (SwiGLU)
            # w1: [hidden, dim]
            # w3: [hidden, dim]
            hb = np.dot(self.w1[l], xb)
            hb2 = np.dot(self.w3[l], xb)
            
            # SwiGLU: x * sigmoid(x) * y
            # hb = hb * sigmoid(hb) * hb2
            
            # sigmoid
            sig = 1.0 / (1.0 + np.exp(-hb))
            hb = hb * sig * hb2
            
            # w2: [dim, hidden]
            xb = np.dot(self.w2[l], hb)
            
            # Residual 2
            x = x + xb
            
        # Final RMSNorm
        x = self.rmsnorm(x, self.rms_final_weight)
        
        # Classifier
        logits = np.dot(self.token_embedding_table, x)
        return logits

def main():
    if len(sys.argv) < 2:
        print("Usage: python reference_oracle.py <model_path> [steps]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    steps = 256
    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])
        
    model = Transformer(model_path)
    print(f"[Oracle] Model loaded. Config: {model.config.dim} dim", file=sys.stderr)
    
    # State initialization
    # We need a proper way to store KV cache that matches numpy ease
    # (n_layers, seq_len, n_kv_heads, head_size) works
    c = model.config
    head_size = c.dim // c.n_heads
    
    # Use lists for dynamic grow (simulating scratchpad logic but in pythonic way)
    # Actually, let's preallocate to match C behavior
    state = {
        'key_cache': np.zeros((c.n_layers, c.seq_len, c.n_kv_heads, head_size), dtype=np.float32),
        'value_cache': np.zeros((c.n_layers, c.seq_len, c.n_kv_heads, head_size), dtype=np.float32)
    }
    
    token = 1 # BOS
    pos = 0
    
    print("[Oracle] Starting inference...", file=sys.stderr)
    
    import time
    t0 = time.time()
    
    while pos < steps:
        logits = model.forward(token, pos, state)
        
        # Argmax
        next_token = np.argmax(logits)
        
        # Output <step> <token>
        print(f"{pos} {token}")
        sys.stdout.flush()
        
        token = next_token
        pos += 1
        
    t1 = time.time()
    dt = t1 - t0
    print(f"[Oracle] Finished {steps} steps in {dt:.4f} s ({steps/dt:.2f} tok/s)", file=sys.stderr)
    
    # Log to benchmarks.csv for figure generation
    with open("benchmarks.csv", "a") as f:
        import datetime
        ts = datetime.datetime.now().ctime()
        # Format: Timestamp,Implementation,Threads,Steps,Time,Tok/s
        f.write(f"{ts},Py_Oracle,1,{steps},{dt:.4f},{steps/dt:.2f}\n")


if __name__ == "__main__":
    main()
