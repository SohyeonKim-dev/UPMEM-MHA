#include "mha_common.h"

#include <mram.h>
#include <alloc.h>
#include <defs.h>
#include <barrier.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit int8_t DPU_Q[NUM_HEADS * SEQ_LEN * HEAD_DIM];
__mram_noinit int8_t DPU_K[NUM_HEADS * SEQ_LEN * HEAD_DIM];
__mram_noinit int8_t DPU_V[NUM_HEADS * SEQ_LEN * HEAD_DIM];

__mram_noinit uint8_t DPU_EXP_LUT[256];
__mram_noinit mha_results_t DPU_RESULTS;

BARRIER_INIT(my_barrier, NR_TASKLETS);

#define TILE_ROWS 8

unsigned int dpu_custom_abs(int32_t x) { 
  return x < 0 ? -x : x; 
}

void dpu_matmul_score_tile(const int8_t* q_tile, const int8_t* k_full, int32_t* score_tile, int tile_rows, int seq_len, int dim) {
    for (int i = 0; i < tile_rows; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            int32_t s = 0;
            for (int d = 0; d < dim; ++d)
                s += (int32_t)q_tile[i*dim + d] * (int32_t)k_full[j*dim + d];
            score_tile[i*seq_len + j] = s;
        }
    }
}

void dpu_softmax_tile(int32_t* score_tile, uint8_t* out_tile, int rows, int cols, uint8_t* lut) {
    for (int i = 0; i < rows; ++i) {
        int32_t row_max = score_tile[i*cols];
        for (int j = 1; j < cols; ++j)
            if (score_tile[i*cols + j] > row_max) row_max = score_tile[i*cols + j];

        int32_t sum = 0;
        uint8_t tmp[SEQ_LEN] __attribute__((aligned(8)));
        for (int j = 0; j < cols; ++j) {
            int32_t val = score_tile[i*cols + j] - row_max;
            int idx = val + 128;
            if (idx < 0) idx = 0;
            if (idx > 255) idx = 255;
            tmp[j] = lut[idx];
            sum += tmp[j];
        }
        for (int j = 0; j < cols; ++j)
            out_tile[i*cols + j] = (uint8_t)((tmp[j] * 255) / (sum ? sum : 1));
    }
}

void dpu_attention_output_tile(const uint8_t* score_tile, const int8_t* v_full, int32_t* out_tile, int tile_rows, int seq_len, int dim) {
    for (int i = 0; i < tile_rows; ++i) {
        for (int d = 0; d < dim; ++d) {
            int32_t s = 0;
            for (int j = 0; j < seq_len; ++j)
                s += (int32_t)score_tile[i*seq_len + j] * (int32_t)v_full[j*dim + d];
            out_tile[i*dim + d] = s;
        }
    }
}

int main(void) {
    unsigned int tid = me();
    if (tid == 0) mem_reset();
    barrier_wait(&my_barrier);

    perfcounter_config(COUNT_CYCLES, true); 

    for (int head = tid; head < NUM_HEADS; head += NR_TASKLETS) {
        perfcounter_config(COUNT_CYCLES, true);

        int32_t score_tile[TILE_ROWS * SEQ_LEN] __attribute__((aligned(8)));
        uint8_t score_u8_tile[TILE_ROWS * SEQ_LEN] __attribute__((aligned(8)));
        int32_t attn_out_tile[TILE_ROWS * HEAD_DIM] __attribute__((aligned(8)));
        int8_t q_tile[TILE_ROWS * HEAD_DIM] __attribute__((aligned(8)));
        int8_t k_full[SEQ_LEN * HEAD_DIM] __attribute__((aligned(8)));
        int8_t v_full[SEQ_LEN * HEAD_DIM] __attribute__((aligned(8)));
        uint8_t exp_lut_cache[256] __attribute__((aligned(8)));

        mram_read((__mram_ptr void const*)(DPU_EXP_LUT), exp_lut_cache, 256);
        mram_read((__mram_ptr void const*)(DPU_K + head * SEQ_LEN * HEAD_DIM),
                  k_full, SEQ_LEN * HEAD_DIM);
        mram_read((__mram_ptr void const*)(DPU_V + head * SEQ_LEN * HEAD_DIM),
                  v_full, SEQ_LEN * HEAD_DIM);

        for (int row_start = 0; row_start < SEQ_LEN; row_start += TILE_ROWS) {
            int rows = (row_start + TILE_ROWS > SEQ_LEN) ? (SEQ_LEN - row_start) : TILE_ROWS;

            mram_read((__mram_ptr void const*)(DPU_Q + head*SEQ_LEN*HEAD_DIM + row_start*HEAD_DIM),
                      q_tile, rows * HEAD_DIM);

            dpu_matmul_score_tile(q_tile, k_full, score_tile, rows, SEQ_LEN, HEAD_DIM);
            dpu_softmax_tile(score_tile, score_u8_tile, rows, SEQ_LEN, exp_lut_cache);
            dpu_attention_output_tile(score_u8_tile, v_full, attn_out_tile, rows, SEQ_LEN, HEAD_DIM);

            size_t bytes = rows * HEAD_DIM * sizeof(int32_t);
            mram_write(attn_out_tile,
                       (__mram_ptr void*)(&DPU_RESULTS.heads[head].out[row_start * HEAD_DIM]),
                       bytes);

            if (row_start == 0) {
                uint64_t cyc = perfcounter_get();
                mram_write(&cyc, (__mram_ptr void*)(&DPU_RESULTS.heads[head].cycles), sizeof(uint64_t));
            }
        }
    }
    return 0;
}
