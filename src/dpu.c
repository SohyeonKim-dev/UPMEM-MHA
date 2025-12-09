#include <stdint.h>
#include <string.h>

#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <perfcounter.h>

#include "common.h"

__mram_noinit int8_t DPU_Q[SLOTS_PER_DPU * SEQ_LEN * HEAD_DIM];
__mram_noinit int8_t DPU_K[SLOTS_PER_DPU * SEQ_LEN * HEAD_DIM];
__mram_noinit int8_t DPU_V[SLOTS_PER_DPU * SEQ_LEN * HEAD_DIM];

__mram_noinit uint8_t DPU_EXP_LUT[256];
__mram_noinit mha_result_t DPU_RESULTS[SLOTS_PER_DPU];

__mram_noinit uint64_t DPU_NHEADS64;
__mram_noinit uint64_t DPU_HEAD0_64;

BARRIER_INIT(my_barrier, NR_TASKLETS);

static inline size_t round_up8(size_t x) { return (x + 7) & ~((size_t)7); }

static int8_t K_shared[SEQ_LEN * HEAD_DIM] __attribute__((aligned(8)));
static int8_t V_shared[SEQ_LEN * HEAD_DIM] __attribute__((aligned(8)));
static uint8_t LUT_shared[256] __attribute__((aligned(8)));

#ifndef Q_BLOCK_ROWS
#define Q_BLOCK_ROWS 8
#endif

void dpu_matmul_score_row(const int8_t *q_row, const int8_t *k_full, int32_t *score_row, int seq_len, int dim) {
    for (int j = 0; j < seq_len; ++j) {
        const int8_t *kv = k_full + (size_t)j * dim;
        int32_t acc = 0;
#pragma unroll 4
        for (int d = 0; d < dim; ++d)
            acc += (int32_t)q_row[d] * (int32_t)kv[d];
        score_row[j] = acc;
    }
}

void dpu_softmax_row(int32_t *score_row, uint8_t *out_row, int cols, const uint8_t *lut) {
    int32_t row_max = score_row[0];
    for (int j = 1; j < cols; ++j)
        if (score_row[j] > row_max) row_max = score_row[j];

    int32_t sum = 0;
    uint8_t tmp[SEQ_LEN] __attribute__((aligned(8)));
    for (int j = 0; j < cols; ++j) {
        int32_t v = score_row[j] - row_max;
        int idx = v + 128;
        if (idx & ~255) idx = (idx < 0) ? 0 : 255;
        uint8_t e = lut[idx];
        tmp[j] = e;
        sum += e;
    }
    if (sum == 0) sum = 1;
    for (int j = 0; j < cols; ++j)
        out_row[j] = (uint8_t)((tmp[j] * 255) / sum);
}

void dpu_attention_output_row(const uint8_t *score_row, const int8_t *v_full, int32_t *out_row, int seq_len, int dim) {
    for (int d = 0; d < dim; ++d) out_row[d] = 0;

    for (int j = 0; j < seq_len; ++j) {
        int32_t s = (int32_t)score_row[j];
        const int8_t *vrow = v_full + (size_t)j * dim;
#pragma unroll 4        
        for (int d = 0; d < dim; ++d) {
            out_row[d] += s * (int32_t)vrow[d];
        }
    }
}

int main(void) {
    unsigned int tid = me();

    if (tid == 0) mem_reset();
    barrier_wait(&my_barrier);

    int32_t score_row[SEQ_LEN] __attribute__((aligned(8)));
    uint8_t score_u8_row[SEQ_LEN] __attribute__((aligned(8)));
    int32_t attn_out_row[HEAD_DIM] __attribute__((aligned(8)));

    int32_t out_padded_local[(HEAD_DIM * sizeof(int32_t) + 7) / 4] __attribute__((aligned(8)));
    int8_t q_block[Q_BLOCK_ROWS * HEAD_DIM] __attribute__((aligned(8)));

    uint64_t nslots64 = 0, slot0_64 = 0;

    if (tid == 0) {
        mram_read((__mram_ptr void*)&DPU_NHEADS64, &nslots64, sizeof(uint64_t));
        mram_read((__mram_ptr void*)&DPU_HEAD0_64, &slot0_64, sizeof(uint64_t));
    }

    barrier_wait(&my_barrier);

    uint32_t nslots = (uint32_t)(nslots64 & 0xffffffffu);
    uint32_t slot0  = (uint32_t)(slot0_64 & 0xffffffffu);
    
    if (nslots == 0) {
        if (tid == 0) {
            uint64_t cyc = perfcounter_get();
            mram_write(&cyc, (__mram_ptr void*)(&DPU_RESULTS[0].cycles), sizeof(uint64_t));
        }
        return 0;
    }

    if (tid == 0) {
        mram_read((__mram_ptr void*)DPU_EXP_LUT, LUT_shared, 256);
    }
    barrier_wait(&my_barrier);

    if (tid == 0) perfcounter_config(COUNT_CYCLES, true);
    barrier_wait(&my_barrier);

    const size_t slot_elems = (size_t)SEQ_LEN * HEAD_DIM;
    const size_t kv_bytes = slot_elems * sizeof(int8_t);

    for (uint32_t ls = 0; ls < nslots; ++ls) {
        size_t slot_elem_offset = (size_t)ls * slot_elems;

        __mram_ptr int8_t *k_base_mram = (__mram_ptr int8_t*)(DPU_K + slot_elem_offset);
        __mram_ptr int8_t *v_base_mram = (__mram_ptr int8_t*)(DPU_V + slot_elem_offset);
        __mram_ptr int8_t *q_base_mram = (__mram_ptr int8_t*)(DPU_Q + slot_elem_offset);

        if (tid == 0) {
            mram_read((__mram_ptr void const*)k_base_mram, K_shared, kv_bytes);
            mram_read((__mram_ptr void const*)v_base_mram, V_shared, kv_bytes);
        }
        barrier_wait(&my_barrier);

        int rows_per_tasklet = (SEQ_LEN + NR_TASKLETS - 1) / NR_TASKLETS;
        int row_start = tid * rows_per_tasklet;
        int row_end = row_start + rows_per_tasklet;
        if (row_start > SEQ_LEN) row_start = SEQ_LEN;
        if (row_end > SEQ_LEN) row_end = SEQ_LEN;

        size_t bytes = (size_t)HEAD_DIM * sizeof(int32_t);
        size_t bytes_padded = round_up8(bytes);
        size_t ints_padded = bytes_padded / sizeof(int32_t);

        for (int r = row_start; r < row_end; r += Q_BLOCK_ROWS) {
            int this_block = row_end - r;
            if (this_block > Q_BLOCK_ROWS) this_block = Q_BLOCK_ROWS;

            __mram_ptr void const* q_block_ptr = (__mram_ptr void const*)(q_base_mram + (size_t)r * HEAD_DIM);
            mram_read(q_block_ptr, q_block, (size_t)this_block * HEAD_DIM * sizeof(int8_t));

            for (int br = 0; br < this_block; ++br) {
                int row_idx = r + br;
                int8_t *q_row_local = q_block + (size_t)br * HEAD_DIM;

                dpu_matmul_score_row(q_row_local, K_shared, score_row, SEQ_LEN, HEAD_DIM);
                dpu_softmax_row(score_row, score_u8_row, SEQ_LEN, LUT_shared);
                dpu_attention_output_row(score_u8_row, V_shared, attn_out_row, SEQ_LEN, HEAD_DIM);

                for (size_t i = 0; i < ints_padded; ++i) out_padded_local[i] = 0;
                for (size_t i = 0; i < (size_t)HEAD_DIM; ++i) out_padded_local[i] = attn_out_row[i];

                __mram_ptr void *out_ptr = (__mram_ptr void*)(&DPU_RESULTS[ls].out[(size_t)row_idx * HEAD_DIM]);
                mram_write(out_padded_local, out_ptr, bytes_padded);
            }
        }
        barrier_wait(&my_barrier);
    }

    if (tid == 0) {
        uint64_t cyc = perfcounter_get();
        for (uint32_t ls = 0; ls < nslots; ++ls)
            mram_write(&cyc, (__mram_ptr void*)(&DPU_RESULTS[ls].cycles), sizeof(uint64_t));
    }
    return 0;
}
