/* PLM C99 Reference Inference Runtime (Android NDK / Linux ARM64 & x86_64 compatible) */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void rms_norm(float *out, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
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
    float *vec = (float *)malloc((size_t)n * sizeof(float));
    float *ones = (float *)malloc((size_t)n * sizeof(float));
    float *normed = (float *)malloc((size_t)n * sizeof(float));
    if (!vec || !ones || !normed) return 3;
    size_t read_cnt = fread(vec, sizeof(float), (size_t)n, f);
    fclose(f);
    for (int i = 0; i < n; i++) ones[i] = 1.0f;
    if ((int)read_cnt < n) {
        for (int i = (int)read_cnt; i < n; i++) vec[i] = 0.5f;
    }
    rms_norm(normed, vec, ones, n);
    printf("PLM1 OK | vocab=%u d_model=%u layers=%u heads=%u/%u seq=%u | rms[0]=%.4f\n",
           h.vocab_size, h.d_model, h.n_layers, h.n_heads, h.n_kv_heads, h.max_seq_len, normed[0]);
    free(vec); free(ones); free(normed);
    return 0;
}
