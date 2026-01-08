/*
 * Offline Quantization Utility (Int4)
 * 
 * Theoretical Basis:
 * This tool performs "Post-Training Quantization" (PTQ) to compress model weights 
 * from FP32 (4 bytes) to Int4 (0.5 bytes). This yields an 8x reduction in 
 * static storage and memory bandwidth requirements.
 * 
 * Format: Group-Wise Symmetric Quantization
 * - Group Size (G): 32. Weights are partitioned into blocks of 32.
 * - Granularity: Each block shares a single FP32 scale factor.
 * - Storage: Two 4-bit weights are packed into a single `uint8_t`.
 * 
 * Trade-off:
 * We trade precision (FP32 -> Int4) for Bandwidth. The choice of G=32 aligns 
 * with the SIMD lane width of AVX2 (8x float) and NEON (4x float unrolled 8x),
 * minimizing "tail effects" during dequantization.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#define GROUP_SIZE 32

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} __attribute__((packed)) Config;

/*
 * Core Quantization Routine
 * 
 * Algorithm: Symmetric Min-Max Quantization
 * Given a group of weights W_g:
 * 1. Find absolute maximum: alpha = max(|w| for w in W_g)
 * 2. Compute Scale: S = alpha / 7.0 (mapping range to [-7, +7])
 * 3. Quantize: w_q = round(w / S)
 * 4. Clip: w_q = clamp(w_q, -7, 7)
 * 5. Offset: w_stored = w_q + 8 (mapping [-7, 7] to [1, 15] for unsigned storage)
 * 
 * Packing Strategy:
 * We store two weights (nibbles) per byte to maximize density.
 * - Low Nibble (bits 0-3): w[2i]
 * - High Nibble (bits 4-7): w[2i+1]
 */
void quantize_buffer(const float* input, uint8_t* out_packed, float* out_scales, int n) {
    int num_groups = n / GROUP_SIZE;
    
    for (int g = 0; g < num_groups; g++) {
        const float* group_data = input + g * GROUP_SIZE;
        
        // Find scale
        float max_val = 0.0f;
        for (int i = 0; i < GROUP_SIZE; i++) {
            float val = fabsf(group_data[i]);
            if (val > max_val) max_val = val;
        }
        
        float scale = max_val / 7.0f;
        if (scale == 0.0f) scale = 1.0f; // Avoid NaN
        out_scales[g] = scale;
        float inv_scale = 1.0f / scale;
        
        // Quantize and Pack
        // packing 2 weights per byte.
        // We iterate 0..31.
        // byte 0: w[0], w[1]
        // ...
        // byte 15: w[30], w[31]
        
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            float w0 = group_data[i];
            float w1 = group_data[i+1];
            
            int8_t q0 = (int8_t)roundf(w0 * inv_scale);
            int8_t q1 = (int8_t)roundf(w1 * inv_scale);
            
            // Clip -7 to 7
            if (q0 < -7) q0 = -7; if (q0 > 7) q0 = 7;
            if (q1 < -7) q1 = -7; if (q1 > 7) q1 = 7;
            
            // Offset to 0..15 for packing (optional, but easier to debug)
            // Or keep as raw bits? 
            // The python script did: w_q + 8. So -7->1, 0->8, 7->15.
            // Let's match Python for consistency.
            uint8_t u0 = (uint8_t)(q0 + 8);
            uint8_t u1 = (uint8_t)(q1 + 8);
            
            // Low nibble = even index (w0)
            // High nibble = odd index (w1)
            uint8_t packed = (u0 & 0xF) | ((u1 & 0xF) << 4);
            
            out_packed[(g * GROUP_SIZE + i) / 2] = packed;
        }
    }
}

// Helper to write buffer
void write_buffer(int fd, const void* buf, size_t size) {
    if (write(fd, buf, size) != size) {
        perror("write");
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    int fd_in = open(argv[1], O_RDONLY);
    if (fd_in == -1) { perror("open input"); return 1; }
    
    size_t file_size = lseek(fd_in, 0, SEEK_END);
    lseek(fd_in, 0, SEEK_SET);
    
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd_in, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    
    int fd_out = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_out == -1) { perror("open output"); return 1; }
    
    // Read Config
    Config* config = (Config*)data;
    printf("Config: dim=%d, layers=%d\n", config->dim, config->n_layers);
    
    write_buffer(fd_out, config, sizeof(Config));
    
    float* weights = (float*)((char*)data + sizeof(Config));
    float* ptr = weights;
    
    // Order:
    // 1. token_embedding_table (vocab * dim) -> Quantize
    // 2. rms_att_weight
    // 3. wq
    // 4. wk
    // 5. wv
    // 6. wo
    // 7. rms_ffn_weight
    // 8. w1
    // 9. w2
    // 10. w3
    // 11. rms_final_weight
    // 12. freq_cis_real
    // 13. freq_cis_imag

    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int n_layers = config->n_layers;
    int n_heads = config->n_heads;
    int n_kv_heads = config->n_kv_heads;
    int vocab_size = abs(config->vocab_size);
    int seq_len = config->seq_len;
    int head_size = dim / n_heads;
    int kv_dim = (dim * n_kv_heads) / n_heads;

    size_t total_written = sizeof(Config);

    // Helper macro to handle quantization or copy
    #define PROCESS_TENSOR(count, do_quantize) do { \
        size_t n = (size_t)(count); \
        if (do_quantize) { \
            uint8_t* packed = malloc(n / 2); \
            float* scales = malloc((n / GROUP_SIZE) * sizeof(float)); \
            quantize_buffer(ptr, packed, scales, n); \
            write_buffer(fd_out, scales, (n / GROUP_SIZE) * sizeof(float)); \
            write_buffer(fd_out, packed, n / 2); \
            free(packed); free(scales); \
            total_written += (n / GROUP_SIZE) * sizeof(float) + n / 2; \
            printf("Quantized tensor size %zu -> %zu\n", n * 4, (n/2) + (n/GROUP_SIZE)*4); \
        } else { \
            write_buffer(fd_out, ptr, n * sizeof(float)); \
            total_written += n * sizeof(float); \
        } \
        ptr += n; \
    } while(0)

    // 1. Token Embed
    PROCESS_TENSOR(vocab_size * dim, 1);
    
    // 2. RMS Att
    PROCESS_TENSOR(n_layers * dim, 0);
    
    // 3. WQ
    PROCESS_TENSOR(n_layers * dim * dim, 1);
    
    // 4. WK
    PROCESS_TENSOR(n_layers * dim * kv_dim, 1);
    
    // 5. WV
    PROCESS_TENSOR(n_layers * dim * kv_dim, 1);
    
    // 6. WO
    PROCESS_TENSOR(n_layers * dim * dim, 1);
    
    // 7. RMS FFN
    PROCESS_TENSOR(n_layers * dim, 0);
    
    // 8. W1
    PROCESS_TENSOR(n_layers * dim * hidden_dim, 1);
    
    // 9. W2
    PROCESS_TENSOR(n_layers * hidden_dim * dim, 1);
    
    // 10. W3
    PROCESS_TENSOR(n_layers * dim * hidden_dim, 1);
    
    // 11. RMS Final
    PROCESS_TENSOR(dim, 0);
    
    // 12. Freq Real
    PROCESS_TENSOR(seq_len * (head_size / 2), 0);
    
    // 13. Freq Imag
    PROCESS_TENSOR(seq_len * (head_size / 2), 0);
    
    printf("Finished. Original size: %zu, New size: %zu. Compression: %.2fx\n", 
           file_size, total_written, (double)file_size / total_written);

    munmap(data, file_size);
    close(fd_in);
    close(fd_out);
    return 0;
}
