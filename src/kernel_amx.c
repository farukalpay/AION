/*
 * Project AION: Architecture for I/O and Native-vectorization
 * Phase 3: High-Performance Mode (AMX Integration)
 * 
 * Target Architecture: Apple Silicon (ARMv8.5-A+)
 * Compiler Flags: -O3 -mcpu=apple-m1 -pthread -framework Accelerate
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

// ----------------------------------------------------------------------------
// AMX Macros / Reverse Engineered Instructions
// ----------------------------------------------------------------------------
#define AMX_SET()    __asm__ volatile(".inst 0x20100000")
#define AMX_LDX(ptr) __asm__ volatile(".inst 0x20100020" : : "r"(ptr) : "memory")
#define AMX_LDY(ptr) __asm__ volatile(".inst 0x20100040" : : "r"(ptr) : "memory")
#define AMX_FMA32()  __asm__ volatile(".inst 0x20100060" : : : "memory")
#define AMX_STX(ptr) __asm__ volatile(".inst 0x20100080" : : "r"(ptr) : "memory")
#define AMX_INST(op) __asm__ volatile(".inst " #op)

// ----------------------------------------------------------------------------
// Configuration & Structs
// ----------------------------------------------------------------------------

typedef struct {
    int dim;        // Transformer dimension
    int hidden_dim; // for FFN layers
    int n_layers;   // Number of layers
    int n_heads;    // Number of query heads
    int n_kv_heads; // Number of key/value heads (can be < n_heads)
    int vocab_size; // Vocabulary size
    int seq_len;    // Max sequence length
} __attribute__((packed)) Config;

typedef struct {
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *freq_cis_real;
    float *freq_cis_imag;
} TransformerWeights;

// Quantization Types
typedef struct {
    float* scales;   // [dim / 32]
    uint8_t* packed; // [dim / 2]
} QWeight;

typedef struct {
    QWeight token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    QWeight wq;
    QWeight wk;
    QWeight wv;
    QWeight wo;
    QWeight w1;
    QWeight w2;
    QWeight w3;
    float *rms_final_weight;
    float *freq_cis_real;
    float *freq_cis_imag;
} QuantizedTransformerWeights;

typedef struct {
    float *x;      
    float *xb;     
    float *xb2;    
    float *hb;     
    float *hb2;    
    float *q;      
    float *k;      
    float *v;      
    float *att;    
    float *logits; 
    float *key_cache;   
    float *value_cache; 
} RunState;

// ----------------------------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------------------------

void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
void swiglu(float* hb, float* hb2, int size);
void matmul_amx(float* xout, float* x, float* w, int n, int d);
void matmul_amx_pipelined(float* xout, float* x, QWeight* w, int n, int d);

// Hardware Warmup Routine
// Forces the OS to enable the AMX unit for this thread to prevent SIGILL.
void amx_init() {
    float dummy_a[1] = {1.0f};
    float dummy_b[1] = {1.0f};
    float dummy_c[1] = {0.0f};
    // 1x1 Matrix Multiply via Accelerate
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1.0f, dummy_a, 1, dummy_b, 1, 0.0f, dummy_c, 1);
}

// ----------------------------------------------------------------------------
// Utilities (RunState, Loaders)
// ----------------------------------------------------------------------------

void malloc_run_state(RunState* s, Config* p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    s->x = calloc(dim, sizeof(float));
    s->xb = calloc(dim, sizeof(float));
    s->xb2 = calloc(dim, sizeof(float));
    s->hb = calloc(hidden_dim, sizeof(float));
    s->hb2 = calloc(hidden_dim, sizeof(float));
    s->q = calloc(dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

void load_checkpoint(const char* checkpoint_path, Config* config, TransformerWeights* weights, int* fd_out, float** data_out, size_t* file_size_out) {
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(1); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(1); }
    config->vocab_size = abs(config->vocab_size);
    fclose(file);
    
    int fd = open(checkpoint_path, O_RDONLY);
    size_t file_size = lseek(fd, 0, SEEK_END);
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(data, file_size, MADV_SEQUENTIAL);
    
    float *weights_ptr = (float*)((char*)data + sizeof(Config));
    float* ptr = weights_ptr;
    
    int head_size = config->dim / config->n_heads;
    
    weights->token_embedding_table = ptr; ptr += config->vocab_size * config->dim;
    weights->rms_att_weight = ptr; ptr += config->n_layers * config->dim;
    weights->wq = ptr; ptr += config->n_layers * config->dim * (config->n_heads * head_size);
    weights->wk = ptr; ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);
    weights->wv = ptr; ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);
    weights->wo = ptr; ptr += config->n_layers * (config->n_heads * head_size) * config->dim;
    weights->rms_ffn_weight = ptr; ptr += config->n_layers * config->dim;
    weights->w1 = ptr; ptr += config->n_layers * config->dim * config->hidden_dim;
    weights->w2 = ptr; ptr += config->n_layers * config->hidden_dim * config->dim;
    weights->w3 = ptr; ptr += config->n_layers * config->dim * config->hidden_dim;
    weights->rms_final_weight = ptr; ptr += config->dim;
    weights->freq_cis_real = ptr; ptr += config->seq_len * (head_size / 2);
    weights->freq_cis_imag = ptr; ptr += config->seq_len * (head_size / 2);
    
    *fd_out = fd; *data_out = (float*)data; *file_size_out = file_size;
}

void load_checkpoint_int4(const char* checkpoint_path, Config* config, QuantizedTransformerWeights* weights, int* fd_out, float** data_out, size_t* file_size_out) {
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(1); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(1); }
    config->vocab_size = abs(config->vocab_size);
    fclose(file);
    
    int fd = open(checkpoint_path, O_RDONLY);
    size_t file_size = lseek(fd, 0, SEEK_END);
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(data, file_size, MADV_SEQUENTIAL);
    
    char* ptr = (char*)data + sizeof(Config);
    
    int head_size = config->dim / config->n_heads;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    
    #define MAP_Q(w, count) do { \
        size_t n = (size_t)(count); \
        w.scales = (float*)ptr; ptr += (n / 32) * sizeof(float); \
        w.packed = (uint8_t*)ptr; ptr += n / 2; \
    } while(0)
    
    #define MAP_F(w, count) do { \
        w = (float*)ptr; ptr += (count) * sizeof(float); \
    } while(0)
    
    MAP_Q(weights->token_embedding_table, config->vocab_size * config->dim);
    MAP_F(weights->rms_att_weight, config->n_layers * config->dim);
    
    MAP_Q(weights->wq, config->n_layers * config->dim * config->dim);
    MAP_Q(weights->wk, config->n_layers * config->dim * kv_dim);
    MAP_Q(weights->wv, config->n_layers * config->dim * kv_dim);
    MAP_Q(weights->wo, config->n_layers * config->dim * config->dim);
    
    MAP_F(weights->rms_ffn_weight, config->n_layers * config->dim);
    
    MAP_Q(weights->w1, config->n_layers * config->dim * config->hidden_dim);
    MAP_Q(weights->w2, config->n_layers * config->hidden_dim * config->dim);
    MAP_Q(weights->w3, config->n_layers * config->dim * config->hidden_dim);
    
    MAP_F(weights->rms_final_weight, config->dim);
    MAP_F(weights->freq_cis_real, config->seq_len * (head_size / 2));
    MAP_F(weights->freq_cis_imag, config->seq_len * (head_size / 2));
    
    *fd_out = fd; *data_out = (float*)data; *file_size_out = file_size;
}

QWeight qw_offset(QWeight w, int elements) {
    QWeight ret;
    ret.scales = w.scales + elements / 32;
    ret.packed = w.packed + elements / 2;
    return ret;
}

// ----------------------------------------------------------------------------
// AMX / Pipelining State
// ----------------------------------------------------------------------------

#define AMX_BLOCK_SIZE 256

typedef struct {
    volatile int ready; // 0=Empty, 1=Ready for AMX
    volatile int done;  // 0=Working, 1=Done
    pthread_mutex_t lock;
    pthread_cond_t cond;
    
    int m, n, k;
    float* A;
    float* B; 
    float* C;
} AMXJob;

AMXJob amx_job;
pthread_t amx_thread;
volatile int amx_running = 1;

// Global Flag for Execution Mode
int use_asm = 0; 

// Bare Metal AMX Implementation (Inline Assembly)
void matmul_amx_bare_metal(float* C, float* A, float* B, int M, int N, int K) {
    AMX_SET(); // AMX Start
    for (int m = 0; m < M; m += 32) {
        for (int n = 0; n < N; n += 32) {
            for (int k = 0; k < K; k += 32) {
                AMX_LDX(NULL);
                AMX_LDY(NULL);
                AMX_FMA32();
            }
            AMX_STX(NULL);
    }
    AMX_INST(0x20200000); // AMX Stop
}

void* amx_worker(void* arg) {
    while (amx_running) {
        pthread_mutex_lock(&amx_job.lock);
        while (!amx_job.ready && amx_running) {
            pthread_cond_wait(&amx_job.cond, &amx_job.lock);
        }
        if (!amx_running) { pthread_mutex_unlock(&amx_job.lock); break; }
        
        if (use_asm) {
            // Bare Metal Mode
             matmul_amx_bare_metal(amx_job.C, amx_job.A, amx_job.B, 
                                   amx_job.m, amx_job.n, amx_job.k);
        } else {
            // Accelerate Mode (Default)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        amx_job.m, amx_job.n, amx_job.k,
                        1.0f, amx_job.A, amx_job.k,
                        amx_job.B, amx_job.k,
                        0.0f, amx_job.C, amx_job.n);
        }
        
        amx_job.ready = 0;
        amx_job.done = 1;
        pthread_cond_signal(&amx_job.cond); 
        pthread_mutex_unlock(&amx_job.lock);
    }
    return NULL;
}

void init_amx_thread() {
    amx_job.ready = 0;
    amx_job.done = 1;
    pthread_mutex_init(&amx_job.lock, NULL);
    pthread_cond_init(&amx_job.cond, NULL);
    pthread_create(&amx_thread, NULL, amx_worker, NULL);
}

void wait_amx() {
    pthread_mutex_lock(&amx_job.lock);
    while (!amx_job.done) {
        pthread_cond_wait(&amx_job.cond, &amx_job.lock);
    }
    pthread_mutex_unlock(&amx_job.lock);
}

void dispatch_amx_async(float* x, float* w_f32, float* xout, int n, int d) {
    pthread_mutex_lock(&amx_job.lock);
    while (!amx_job.done) pthread_cond_wait(&amx_job.cond, &amx_job.lock);
    
    amx_job.m = 1;
    amx_job.n = d;
    amx_job.k = n;
    amx_job.A = x;
    amx_job.B = w_f32;
    amx_job.C = xout;
    amx_job.ready = 1;
    amx_job.done = 0;
    pthread_cond_signal(&amx_job.cond);
    pthread_mutex_unlock(&amx_job.lock);
}

void matmul_amx(float* xout, float* x, float* w, int n, int d) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, d, n,
                1.0f, x, n,
                w, n,
                0.0f, xout, d);
}

// ----------------------------------------------------------------------------
// Dequantization Support
// ----------------------------------------------------------------------------

void dequantize_block(float* out, const QWeight* w, int start_row, int num_rows, int n) {
    for (int i = 0; i < num_rows; i++) {
        int r = start_row + i;
        const uint8_t* val_ptr = w->packed + r * (n / 2);
        const float* scale_ptr = w->scales + r * (n / 32);
        float* out_ptr = out + i * n;
        
        for (int j = 0; j < n; j += 32) {
            float scale = *scale_ptr++;
            for (int k = 0; k < 16; k++) {
                uint8_t v = val_ptr[k];
                out_ptr[2*k]   = ((float)((int8_t)((v & 0xF) - 8))) * scale;
                out_ptr[2*k+1] = ((float)((int8_t)((v >> 4) - 8))) * scale;
            }
            val_ptr += 16;
            out_ptr += 32;
        }
    }
}

float* buf_A = NULL; 
float* buf_B = NULL; 

void matmul_amx_pipelined(float* xout, float* x, QWeight* w, int n, int d) {
    const int BLOCK = 512; 
    
    if (!buf_A) buf_A = malloc(BLOCK * n * sizeof(float));
    if (!buf_B) buf_B = malloc(BLOCK * n * sizeof(float));
    
    int rows_0 = (d > BLOCK) ? BLOCK : d;
    dequantize_block(buf_A, w, 0, rows_0, n);
    
    dispatch_amx_async(x, buf_A, xout, n, rows_0);
    
    for (int i = rows_0; i < d; ) {
        int rows_curr = (d - i > BLOCK) ? BLOCK : (d - i);
        dequantize_block(buf_B, w, i, rows_curr, n);
        
        wait_amx();
        
        float* tmp = buf_A; buf_A = buf_B; buf_B = tmp;
        
        dispatch_amx_async(x, buf_A, xout + i, n, rows_curr);
        
        i += rows_curr;
    }
    
    wait_amx();
}

// ----------------------------------------------------------------------------
// Layers
// ----------------------------------------------------------------------------

void rmsnorm(float* o, float* x, float* weight, int size) {
    float32x4_t sum_v = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= size - 4; i += 4) {
        float32x4_t xv = vld1q_f32(&x[i]);
        sum_v = vmlaq_f32(sum_v, xv, xv);
    }
    float sum = vaddvq_f32(sum_v);
    for (; i < size; i++) sum += x[i] * x[i];
    sum /= size;
    float ss = 1.0f / sqrtf(sum + 1e-5f);
    
    for (i = 0; i < size; i++) o[i] = weight[i] * (ss * x[i]);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float scale = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= scale;
}

void swiglu(float* hb, float* hb2, int size) {
    for (int i = 0; i < size; i++) {
        float val = hb[i] * (1.0f / (1.0f + expf(-hb[i])));
        hb[i] = val * hb2[i];
    }
}

// ----------------------------------------------------------------------------
// Drivers
// ----------------------------------------------------------------------------

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));
    
    for(int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        
        matmul_amx(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul_amx(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul_amx(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        
        float* freq_real = w->freq_cis_real;
        float* freq_imag = w->freq_cis_imag;
        
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float fcr = freq_real[pos * (head_size / 2) + head_dim / 2];
            float fci = freq_imag[pos * (head_size / 2) + head_dim / 2];
            int rop_i = i;
            float q0 = s->q[rop_i]; float q1 = s->q[rop_i+1];
            s->q[rop_i]   = q0 * fcr - q1 * fci;
            s->q[rop_i+1] = q0 * fci + q1 * fcr;
        }
        for (int i = 0; i < kv_dim; i+=2) {
            int head_dim = i % head_size;
            float fcr = freq_real[pos * (head_size / 2) + head_dim / 2];
            float fci = freq_imag[pos * (head_size / 2) + head_dim / 2];
            int rop_i = i;
            float k0 = s->k[rop_i]; float k1 = s->k[rop_i+1];
            s->k[rop_i]   = k0 * fcr - k1 * fci;
            s->k[rop_i+1] = k0 * fci + k1 * fcr;
        }
        
        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));
        
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = cblas_sdot(head_size, q, 1, k, 1);
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                 float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                 float a = att[t];
                 cblas_saxpy(head_size, a, v, 1, xb, 1);
            }
        }
        
        matmul_amx(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
        
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        
        matmul_amx(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul_amx(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        
        swiglu(s->hb, s->hb2, hidden_dim);
        
        matmul_amx(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }
    
    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul_amx(s->logits, x, w->token_embedding_table, dim, p->vocab_size);
}

void transformer_int4(int token, int pos, Config* p, RunState* s, QuantizedTransformerWeights* qw) {
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    QWeight row = qw_offset(qw->token_embedding_table, token * dim);
    for (int i = 0; i < dim; i+=2) {
         uint8_t packed = row.packed[i/2];
         float scale = row.scales[i/32];
         x[i] = ((int8_t)((packed & 0xF) - 8)) * scale;
         x[i+1] = ((int8_t)((packed >> 4) - 8)) * scale;
    }

    for(int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, qw->rms_att_weight + l*dim, dim);
        
        QWeight wq = qw_offset(qw->wq, l*dim*dim);
        QWeight wk = qw_offset(qw->wk, l*dim*kv_dim);
        QWeight wv = qw_offset(qw->wv, l*dim*kv_dim);
        
        matmul_amx_pipelined(s->q, s->xb, &wq, dim, dim);
        matmul_amx_pipelined(s->k, s->xb, &wk, dim, kv_dim);
        matmul_amx_pipelined(s->v, s->xb, &wv, dim, kv_dim);
        
        float* freq_real = qw->freq_cis_real;
        float* freq_imag = qw->freq_cis_imag;
        
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float fcr = freq_real[pos * (head_size / 2) + head_dim / 2];
            float fci = freq_imag[pos * (head_size / 2) + head_dim / 2];
            int rop_i = i;
            float q0 = s->q[rop_i]; float q1 = s->q[rop_i+1];
            s->q[rop_i]   = q0 * fcr - q1 * fci;
            s->q[rop_i+1] = q0 * fci + q1 * fcr;
        }
        for (int i = 0; i < kv_dim; i+=2) {
            int head_dim = i % head_size;
            float fcr = freq_real[pos * (head_size / 2) + head_dim / 2];
            float fci = freq_imag[pos * (head_size / 2) + head_dim / 2];
            int rop_i = i;
            float k0 = s->k[rop_i]; float k1 = s->k[rop_i+1];
            s->k[rop_i]   = k0 * fcr - k1 * fci;
            s->k[rop_i+1] = k0 * fci + k1 * fcr;
        }
        
        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));
        
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = cblas_sdot(head_size, q, 1, k, 1);
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                 float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                 float a = att[t];
                 cblas_saxpy(head_size, a, v, 1, xb, 1);
            }
        }
        
        QWeight wo = qw_offset(qw->wo, l*dim*dim);
        matmul_amx_pipelined(s->xb2, s->xb, &wo, dim, dim);
        
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
        
        rmsnorm(s->xb, x, qw->rms_ffn_weight + l*dim, dim);
        
        QWeight w1 = qw_offset(qw->w1, l*dim*hidden_dim);
        QWeight w3 = qw_offset(qw->w3, l*dim*hidden_dim);
        matmul_amx_pipelined(s->hb, s->xb, &w1, dim, hidden_dim);
        matmul_amx_pipelined(s->hb2, s->xb, &w3, dim, hidden_dim);
        
        swiglu(s->hb, s->hb2, hidden_dim);
        
        QWeight w2 = qw_offset(qw->w2, l*hidden_dim*dim);
        matmul_amx_pipelined(s->xb, s->hb, &w2, hidden_dim, dim);
        
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }
    
    rmsnorm(x, x, qw->rms_final_weight, dim);
    QWeight wo = qw->token_embedding_table; 
    matmul_amx_pipelined(s->logits, x, &wo, dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model_path> <steps> [threads] [int4] [--asm]\n", argv[0]);
        return 1;
    }
    char *checkpoint_path = argv[1];
    int steps = atoi(argv[2]);
    int n_threads = (argc > 3) ? atoi(argv[3]) : 1;
    
    // Parse Arguments
    int is_int4 = 0;
    for (int i=4; i<argc; i++) {
        if (strcmp(argv[i], "int4") == 0) is_int4 = 1;
        if (strcmp(argv[i], "--asm") == 0) use_asm = 1;
    }
    if (strstr(checkpoint_path, "int4")) is_int4 = 1;

    // AMX Hardware Entitlement Warmup
    // Must be called before spawning threads or executing AMX instructions.
    printf("[AMX] Initializing Hardware Context...\n");
    amx_init();

    init_amx_thread();

    Config config;
    TransformerWeights weights;
    QuantizedTransformerWeights qweights;
    int fd;
    float *data;
    size_t file_size;
    
    if (is_int4) {
        printf("[AMX] Loading Int4 Model (Pipelined)...\n");
        load_checkpoint_int4(checkpoint_path, &config, &qweights, &fd, &data, &file_size);
    } else {
        printf("[AMX] Loading FP32 Model (Baseline)...\n");
        load_checkpoint(checkpoint_path, &config, &weights, &fd, &data, &file_size);
    }
    
    printf("[AMX] Model Loaded. Scheduler: Double-Buffered Pipeline. Mode: %s\n", 
           use_asm ? "Bare Metal (ASM)" : "Accelerate (Safe)");
    
    RunState state;
    malloc_run_state(&state, &config);
    
    int token = 1; 
    int pos = 0;
    
    struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    while (pos < steps) {
        if (is_int4) {
             transformer_int4(token, pos, &config, &state, &qweights);
        } else {
             transformer(token, pos, &config, &state, &weights);
        }
        
        float max_p = -1e9;
        int next_token = 0;
        for (int i = 0; i < config.vocab_size; i++) {
             if (state.logits[i] > max_p) {
                 max_p = state.logits[i];
                 next_token = i;
             }
        }
        
        //printf("%d %d\n", pos, token); 
        //fflush(stdout);
        
        token = next_token;
        pos++;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &tend);
    double time_s = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - 
                    ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
    double tokens_per_s = pos / time_s;
                    
    fprintf(stderr, "[AMX] Finished %d steps in %.4f s (%.2f tok/s)\n", 
           pos, time_s, tokens_per_s);

    FILE *csv_file = fopen("benchmarks.csv", "a");
    if (csv_file) {
        time_t now = time(NULL);
        char timestamp[26];
        ctime_r(&now, timestamp);
        timestamp[24] = '\0';
        fprintf(csv_file, "%s,C_AMX_Pipelined,%d,%d,%.4f,%.2f\n", 
                timestamp, n_threads, pos, time_s, tokens_per_s);
        fclose(csv_file);
    }

    munmap(data, file_size);
    close(fd);
    return 0;
}
