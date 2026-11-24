/* PLM v0.3 C99 Reference Inference Runtime
 * Portable ARM64 (Android NDK / Termux) & x86_64 CPU kernel:
 * - PLM1 binary header loader
 * - RMSNorm
 * - Block-wise Q4_0 dequantization & quantized matrix-vector multiply (q4_matvec)
 * - Softmax & CPU latency benchmark
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q4_BLOCK 32

typedef struct {
    char magic[4];
    uint32_t vocab_size;
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t ffn_hidden;
    uint32_t max_seq_len;
} PlmHeader;

typedef struct {
    float scale;
    int8_t q[Q4_BLOCK]; /* values in [-7, 7] */
} Q4Block;

static void rms_norm(float *out, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static void q4_matvec(float *out, const Q4Block *blocks, const float *x, int rows, int cols) {
    int blocks_per_row = cols / Q4_BLOCK;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const Q4Block *blk = &blocks[r * blocks_per_row + b];
            const float *xp = &x[b * Q4_BLOCK];
            float partial = 0.0f;
            for (int k = 0; k < Q4_BLOCK; k++) {
                partial += (float)blk->q[k] * xp[k];
            }
            acc += partial * blk->scale;
        }
        out[r] = acc;
    }
}

static void softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Hasznalat: %s <model.plm>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("Nem sikerult megnyitni a .plm fajlt");
        return 1;
    }
    PlmHeader h;
    if (fread(&h, sizeof(PlmHeader), 1, f) != 1 || memcmp(h.magic, "PLM1", 4) != 0) {
        fprintf(stderr, "Ervenytelen PLM binaris fejlec!\n");
        fclose(f);
        return 2;
    }

    int n = (int)h.d_model;
    if (n < Q4_BLOCK) n = Q4_BLOCK;
    int cols = (n / Q4_BLOCK) * Q4_BLOCK;
    int rows = 64;
    int bpr = cols / Q4_BLOCK;

    float *vec = (float *)malloc((size_t)cols * sizeof(float));
    float *ones = (float *)malloc((size_t)cols * sizeof(float));
    float *normed = (float *)malloc((size_t)cols * sizeof(float));
    float *out = (float *)malloc((size_t)rows * sizeof(float));
    Q4Block *qmat = (Q4Block *)malloc((size_t)(rows * bpr) * sizeof(Q4Block));
    if (!vec || !ones || !normed || !out || !qmat) {
        fclose(f);
        return 3;
    }

    size_t read_cnt = fread(vec, sizeof(float), (size_t)cols, f);
    fclose(f);
    for (int i = 0; i < cols; i++) {
        ones[i] = 1.0f;
        if ((int)read_cnt <= i) vec[i] = 0.25f;
    }
    for (int i = 0; i < rows * bpr; i++) {
        qmat[i].scale = 0.05f;
        for (int k = 0; k < Q4_BLOCK; k++) {
            qmat[i].q[k] = (int8_t)(((i + k) % 15) - 7);
        }
    }

    rms_norm(normed, vec, ones, cols);

    clock_t t0 = clock();
    for (int iter = 0; iter < 50; iter++) {
        q4_matvec(out, qmat, normed, rows, cols);
    }
    softmax(out, rows);
    double ms = ((double)(clock() - t0) * 1000.0) / (double)CLOCKS_PER_SEC;

    printf("PLM1 OK | vocab=%u d_model=%u layers=%u heads=%u/%u seq=%u | rms[0]=%.4f | q4_prob0=%.4f | q4_bench_ms=%.3f\n",
           h.vocab_size, h.d_model, h.n_layers, h.n_heads, h.n_kv_heads, h.max_seq_len,
           normed[0], out[0], ms);

    free(vec); free(ones); free(normed); free(out); free(qmat);
    return 0;
}
