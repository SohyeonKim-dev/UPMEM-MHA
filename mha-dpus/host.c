#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <time.h>
#include <dpu.h>

#include "common.h"

#ifndef DPU_BINARY
#define DPU_BINARY "dpus.mpo"
#endif

static int8_t input_Q[TOTAL_SLOTS * SEQ_LEN * HEAD_DIM];
static int8_t input_K[TOTAL_SLOTS * SEQ_LEN * HEAD_DIM];
static int8_t input_V[TOTAL_SLOTS * SEQ_LEN * HEAD_DIM];

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
    for (int i = 0; i < len; ++i)
        for (int j = 0; j < len; ++j) {
            int32_t s = 0;
            for (int d = 0; d < dim; ++d) s += (int32_t)q[i*dim + d] * (int32_t)k[j*dim + d];
            score[i*len + j] = s;
        }
}

void host_softmax_int32(int32_t* score, uint8_t* out, int rows, int cols, uint8_t* exp_lut_ptr) {
    for (int i = 0; i < rows; ++i) {
        int32_t row_max = score[i*cols];

        for (int j = 1; j < cols; ++j) {
            if (score[i*cols + j] > row_max) {
                row_max = score[i*cols + j];
            }
        }

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
        for (int j = 0; j < cols; ++j) {
            out[i*cols + j] = (uint8_t)((tmp[j] * 255) / (sum ? sum : 1));
        }
    }
}

void host_attention_output_int8(const uint8_t* score, const int8_t* v, int32_t* out, int len, int dim) {
    for (int i = 0; i < len; ++i) {
        for (int d = 0; d < dim; ++d) {
            int32_t s = 0;
            for (int j = 0; j < len; ++j) {
                s += (int32_t)score[i*len + j] * (int32_t)v[j*dim + d];
            }
            out[i*dim + d] = s;
        }
    }
}

void host_compute_reference() {
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int slot = h * BATCH_SIZE + b;
            int8_t* q = input_Q + (size_t)slot * SEQ_LEN * HEAD_DIM;
            int8_t* k = input_K + (size_t)slot * SEQ_LEN * HEAD_DIM;
            int8_t* v = input_V + (size_t)slot * SEQ_LEN * HEAD_DIM;

            int32_t score[SEQ_LEN * SEQ_LEN];
            uint8_t score_u8[SEQ_LEN * SEQ_LEN];
            int32_t out[SEQ_LEN * HEAD_DIM];

            host_matmul_score_int8(q, k, score, SEQ_LEN, HEAD_DIM);
            host_softmax_int32(score, score_u8, SEQ_LEN, SEQ_LEN, exp_lut);
            host_attention_output_int8(score_u8, v, out, SEQ_LEN, HEAD_DIM);

            for (int i = 0; i < SEQ_LEN * HEAD_DIM; ++i) {
                host_results.heads[slot].out[i] = out[i];
            }
            host_results.heads[slot].cycles = 0;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double host_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    printf("Host total computation time: %.3f ms\n", host_ms);
}

void compare_and_print() {
    bool equal = true;
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int slot = h * BATCH_SIZE + b;
            for (int i = 0; i < SEQ_LEN * HEAD_DIM; ++i) {
                float dpu_val = (float)dpu_results.heads[slot].out[i] / ((float)QK_SCALE * (float)V_SCALE);
                float host_val = (float)host_results.heads[slot].out[i] / ((float)QK_SCALE * (float)V_SCALE);
                float diff = fabs(host_val - dpu_val);
                
                if (diff > 1e-2f) {
                    equal = false;
                }
            }
        }
    }

    printf("\n--- DPU cycles summary ---\n");

    uint64_t total_cycles = 0;
    for (int slot = 0; slot < TOTAL_SLOTS; ++slot) {
        uint64_t c = dpu_results.heads[slot].cycles;
        total_cycles += c;
    }

    double avg_cycles = (double)total_cycles / (double)TOTAL_SLOTS;
    double avg_ms = avg_cycles / 350000.0;

    printf("Total cycles (sum over all slots): %llu\n", (unsigned long long)total_cycles);
    printf("Average cycles per slot: %.0f (%.3f ms)\n", avg_cycles, avg_ms);

    if (equal) {
        printf("Host == DPU\n");
    } else {
        printf("Host != DPU\n");
    }
}

int main(void) {
    struct dpu_set_t set, dpu;
    uint32_t nr_dpus = NR_DPUS; 

    DPU_ASSERT(dpu_alloc(nr_dpus, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    printf("DPUs allocated: %u\n", nr_dpus);

    if (nr_dpus != NR_DPUS) {
        fprintf(stderr, "Error: expected %u DPUs but got %u\n", NR_DPUS, nr_dpus);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    size_t total_elems_per_slot = (size_t)SEQ_LEN * HEAD_DIM;
    size_t total_bytes_per_slot = total_elems_per_slot * sizeof(int8_t);

    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int slot = h * BATCH_SIZE + b;
            init_input_data(input_Q + (size_t)slot * total_elems_per_slot, (int)total_elems_per_slot, 1 + slot);
            init_input_data(input_K + (size_t)slot * total_elems_per_slot, (int)total_elems_per_slot, 100 + slot);
            init_input_data(input_V + (size_t)slot * total_elems_per_slot, (int)total_elems_per_slot, 200 + slot);
        }
    }

    init_exp_lut(exp_lut);

    size_t matrix_size = total_bytes_per_slot; 
    uint32_t slot_idx = 0;
    uint32_t dpu_idx = 0;

    DPU_FOREACH(set, dpu, dpu_idx) {
        uint32_t remaining = TOTAL_SLOTS - slot_idx;
        uint32_t nslots = (remaining >= SLOTS_PER_DPU) ? SLOTS_PER_DPU : remaining;

        size_t bytes = (size_t)nslots * matrix_size;
        DPU_ASSERT(dpu_copy_to(dpu, "DPU_Q", 0, input_Q + (size_t)slot_idx * matrix_size, bytes));
        DPU_ASSERT(dpu_copy_to(dpu, "DPU_K", 0, input_K + (size_t)slot_idx * matrix_size, bytes));
        DPU_ASSERT(dpu_copy_to(dpu, "DPU_V", 0, input_V + (size_t)slot_idx * matrix_size, bytes));

        uint64_t n64 = (uint64_t)nslots;
        uint64_t head0_64 = (uint64_t)slot_idx;
        
        DPU_ASSERT(dpu_copy_to(dpu, "DPU_NHEADS64", 0, &n64, sizeof(uint64_t)));
        DPU_ASSERT(dpu_copy_to(dpu, "DPU_HEAD0_64", 0, &head0_64, sizeof(uint64_t)));

        slot_idx += nslots;
    }

    DPU_ASSERT(dpu_copy_to(set, "DPU_EXP_LUT", 0, exp_lut, sizeof(exp_lut)));
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    dpu_idx = 0;
    slot_idx = 0;
  
    DPU_FOREACH(set, dpu, dpu_idx) {
        uint32_t remaining = TOTAL_SLOTS - slot_idx;
        uint32_t nslots = (remaining >= SLOTS_PER_DPU) ? SLOTS_PER_DPU : remaining;
        size_t bytes = (size_t)nslots * sizeof(mha_result_t);
        DPU_ASSERT(dpu_copy_from(dpu, "DPU_RESULTS", 0, &dpu_results.heads[slot_idx], bytes));
        slot_idx += nslots;
    }

    host_compute_reference();
    compare_and_print();

    DPU_ASSERT(dpu_free(set));
    return 0;
}
