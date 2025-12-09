#include "mha_common.h"

#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <alloc.h>
#include <defs.h>
#include <barrier.h>
#include <perfcounter.h>

__mram_noinit float DPU_Q[SEQ_LEN * HEAD_DIM * NUM_HEADS];
__mram_noinit float DPU_K[SEQ_LEN * HEAD_DIM * NUM_HEADS];
__mram_noinit float DPU_V[SEQ_LEN * HEAD_DIM * NUM_HEADS];

__host mha_results_t DPU_RESULTS;

// 배리어들은 매크로로 정의됨 -> dpu 내부에서 tasklet들 동기화 
BARRIER_INIT(my_barrier, NR_TASKLETS); 

float dpu_custom_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    float output = x;
    for (int i = 0; i < 6; i++) {
        output = 0.5f * (output + x / output);
    }
    return output;
}

float dpu_custom_exp(float x) {
    float result = 1.0f + x + (x*x)*0.5f + (x*x*x)/6.0f;
    return result;
}

float dpu_custom_fabs(float x) {
    return x < 0.0f ? -x : x;
}

void dpu_matmul_score(const float* q, const float* k, float* score, int len, int dim) {
    float scale = 1.0f / dpu_custom_sqrt((float)dim);
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            float s = 0.0f;
            for (int d = 0; d < dim; d++) {
                s += q[i * dim + d] * k[j * dim + d];
            }
            score[i * len + j] = s * scale; // 내적 계산
        }
    }
}

void dpu_softmax(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float maxval = mat[i * cols];
        for (int j = 1; j < cols; j++) {
            if (mat[i * cols + j] > maxval) maxval = mat[i * cols + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = dpu_custom_exp(mat[i * cols + j] - maxval);
            mat[i * cols + j] = val;
            sum += val;
        }
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] /= sum;
        }
    }
}

void dpu_attention_output(const float* score, const float* v, float* out, int len, int dim) {
    for (int i = 0; i < len; i++) {
        for (int d = 0; d < dim; d++) {
            float s = 0.0f;
            for (int j = 0; j < len; j++) {
                s += score[i * len + j] * v[j * dim + d];
            }
            out[i * dim + d] = s;
        }
    }
}

int main() {
    unsigned int tid = me(); // 현재 실행중인 tasklet id 가져오는거 (head 역할)

    if (tid == 0) { // == 0 을 추가로 걸어도, 해결 X / barrier_wait과 순서 바꿔도 X
        mem_reset();
    }

    barrier_wait(&my_barrier); 

    perfcounter_config(COUNT_CYCLES, true);

    // 0부터 시작하는거라 등호 X
    if (tid < NUM_HEADS) { 
        // q k v가 cache(wram) 역할임 ->  mram_read example 에 대응해보면.
        float q[SEQ_LEN * HEAD_DIM];
        float k[SEQ_LEN * HEAD_DIM];
        float v[SEQ_LEN * HEAD_DIM];

        float score[SEQ_LEN * SEQ_LEN];
        float attn_out[SEQ_LEN * HEAD_DIM];

        // * mram_read example 
        // mram_read(&DPU_BUFFER[buffer_idx], cache, BLOCK_SIZE); // mram -> wram (cache) dma 전송 
        // mram_read((__mram_ptr void const*)temp_address_A + (k * cache_size), local_cache_A, cache_size);

        // q k v를 가져오는 과정 (mram에서 가져와서 채워주는거지) -> wram 용량때문에 에러가 난 것
        mram_read((__mram_ptr void const *)(DPU_Q + tid * SEQ_LEN * HEAD_DIM), q, SEQ_LEN * HEAD_DIM * sizeof(float));
        mram_read((__mram_ptr void const *)(DPU_K + tid * SEQ_LEN * HEAD_DIM), k, SEQ_LEN * HEAD_DIM * sizeof(float));
        mram_read((__mram_ptr void const *)(DPU_V + tid * SEQ_LEN * HEAD_DIM), v, SEQ_LEN * HEAD_DIM * sizeof(float));

        dpu_matmul_score(q, k, score, SEQ_LEN, HEAD_DIM);
        dpu_softmax(score, SEQ_LEN, SEQ_LEN);
        dpu_attention_output(score, v, attn_out, SEQ_LEN, HEAD_DIM);

        for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++)
            DPU_RESULTS.heads[tid].out[i] = attn_out[i];

        DPU_RESULTS.heads[tid].cycles = perfcounter_get();
    }
    return 0;
}
