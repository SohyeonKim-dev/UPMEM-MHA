#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mha_common.h"
#include <dpu.h>

#ifndef DPU_BINARY
#define DPU_BINARY "mha_tiling.mpo"
#endif

static int8_t input_Q[NUM_HEADS * SEQ_LEN * HEAD_DIM];
static int8_t input_K[NUM_HEADS * SEQ_LEN * HEAD_DIM];
static int8_t input_V[NUM_HEADS * SEQ_LEN * HEAD_DIM];

static mha_results_t dpu_results;   
static mha_results_t host_results;  

uint8_t exp_lut[256];

void init_exp_lut(uint8_t* lut) {
    for (int i = 0; i < 256; ++i) {
        float x = (float)(i - 128) / 32.0f;
        float e = expf(x);
        if (e > 255.0f) e = 255.0f;
        lut[i] = (uint8_t)e;
    }
}

void init_input_data(int8_t *arr, int size, int seed_offset) {
    srand(42 + seed_offset);
    for (int i = 0; i < size; ++i) {
        float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        arr[i] = (int8_t)(val * QK_SCALE);
    }
}

void host_matmul_score_int8(const int8_t* q, const int8_t* k, int32_t* score, int len, int dim) {
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            int32_t s = 0;
            for (int d = 0; d < dim; ++d) s += (int32_t)q[i*dim + d] * (int32_t)k[j*dim + d];
            score[i*len + j] = s;
        }
    }
}

void host_softmax_int32(int32_t* score, uint8_t* out, int rows, int cols, uint8_t* exp_lut_ptr) {
    for (int i = 0; i < rows; ++i) {
        int32_t row_max = score[i*cols];
        for (int j = 1; j < cols; ++j) if (score[i*cols + j] > row_max) row_max = score[i*cols + j];
        int32_t sum = 0;
        uint8_t tmp[SEQ_LEN];
        for (int j = 0; j < cols; ++j) {
            int32_t val = score[i*cols + j] - row_max;
            int idx = val + 128;
            if (idx < 0) idx = 0;
            if (idx > 255) idx = 255;
            tmp[j] = exp_lut_ptr[idx];
            sum += tmp[j];
        }
        for (int j = 0; j < cols; ++j) out[i*cols + j] = (uint8_t)((tmp[j] * 255) / (sum ? sum : 1));
    }
}

void host_attention_output_int8(const uint8_t* score, const int8_t* v, int32_t* out, int len, int dim) {
    for (int i = 0; i < len; ++i) {
        for (int d = 0; d < dim; ++d) {
            int32_t s = 0;
            for (int j = 0; j < len; ++j) s += (int32_t)score[i*len + j] * (int32_t)v[j*dim + d];
            out[i*dim + d] = s;
        }
    }
}

void host_compute_reference() {
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    for (int h = 0; h < NUM_HEADS; ++h) {
        int8_t* q = input_Q + h * SEQ_LEN * HEAD_DIM;
        int8_t* k = input_K + h * SEQ_LEN * HEAD_DIM;
        int8_t* v = input_V + h * SEQ_LEN * HEAD_DIM;

        int32_t score[SEQ_LEN * SEQ_LEN];
        uint8_t score_u8[SEQ_LEN * SEQ_LEN];
        int32_t out[SEQ_LEN * HEAD_DIM];

        host_matmul_score_int8(q, k, score, SEQ_LEN, HEAD_DIM);
        host_softmax_int32(score, score_u8, SEQ_LEN, SEQ_LEN, exp_lut);
        host_attention_output_int8(score_u8, v, out, SEQ_LEN, HEAD_DIM);

        for (int i = 0; i < SEQ_LEN * HEAD_DIM; ++i)
            host_results.heads[h].out[i] = out[i]; 
        host_results.heads[h].cycles = 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);  
    double host_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    printf("Host total computation time: %.3f ms\n", host_ms);
}

void compare_and_print() {
    bool equal = true;

    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < SEQ_LEN * HEAD_DIM; ++i) {
            float dpu_val = (float)dpu_results.heads[h].out[i] / ((float)QK_SCALE * (float)V_SCALE);
            float host_val = (float)host_results.heads[h].out[i] / ((float)QK_SCALE * (float)V_SCALE);
            float diff = fabs(host_val - dpu_val);
            if (diff > 1e-2f) equal = false;
        }
    }
    printf("\n--- DPU cycles per head ---\n");

    uint64_t total_cycles = 0;
    for (int h = 0; h < NUM_HEADS; ++h) {
        uint64_t c = dpu_results.heads[h].cycles;
        printf("Head %d cycles: %llu (%.3f ms)\n",
               h, (unsigned long long)c, c / 350000.0);
        total_cycles += c;  
    }
    if (equal) printf("Host == DPU\n");
    else printf("Host != DPU\n");
}

int main(void) {
    struct dpu_set_t set, dpu;
    uint32_t nr_dpus = 128;

    DPU_ASSERT(dpu_alloc(nr_dpus, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    printf("DPUs allocated: %u\n", nr_dpus);

    init_input_data(input_Q, NUM_HEADS * SEQ_LEN * HEAD_DIM, 1);
    init_input_data(input_K, NUM_HEADS * SEQ_LEN * HEAD_DIM, 2);
    init_input_data(input_V, NUM_HEADS * SEQ_LEN * HEAD_DIM, 3);
    init_exp_lut(exp_lut);

    size_t matrix_size = SEQ_LEN * HEAD_DIM * sizeof(int8_t);

    for (int h = 0; h < NUM_HEADS; ++h) {
        DPU_ASSERT(dpu_copy_to(set, "DPU_Q", h * matrix_size, input_Q + h * SEQ_LEN * HEAD_DIM, matrix_size));
        DPU_ASSERT(dpu_copy_to(set, "DPU_K", h * matrix_size, input_K + h * SEQ_LEN * HEAD_DIM, matrix_size));
        DPU_ASSERT(dpu_copy_to(set, "DPU_V", h * matrix_size, input_V + h * SEQ_LEN * HEAD_DIM, matrix_size));
    }

    DPU_ASSERT(dpu_copy_to(set, "DPU_EXP_LUT", 0, exp_lut, 256));

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    DPU_ASSERT(dpu_prepare_xfer(set, &dpu_results));
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, sizeof(mha_results_t), DPU_XFER_DEFAULT));

    host_compute_reference();
    compare_and_print();

    DPU_ASSERT(dpu_free(set));
    return 0;
}

