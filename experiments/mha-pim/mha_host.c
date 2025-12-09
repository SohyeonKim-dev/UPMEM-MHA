#include "mha_common.h"

#include <dpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DPU_BINARY
#define DPU_BINARY "mha.mpo"
#endif

// dpu => 각 head(tasklet) 별 수행까지 처리 
// cpu host => 각 head의 결과를 concat 

static float input_Q[SEQ_LEN * HEAD_DIM * NUM_HEADS];
static float input_K[SEQ_LEN * HEAD_DIM * NUM_HEADS];
static float input_V[SEQ_LEN * HEAD_DIM * NUM_HEADS];

static mha_results_t dpu_results;
static mha_results_t host_results;

void init_input_data(float *arr, int size, int seed_offset) {
    srand(42 + seed_offset);

    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;  
    }
}

// dpu와 동일한 로직을 host(cpu)로도 돌려준다 -> 확인용이니까 똑같이.
float host_custom_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    float y = x;
    for (int i = 0; i < 6; i++) {
        y = 0.5f * (y + x / y);
    }
    return y;
}

float host_custom_exp(float x) {
    return 1.0f + x + (x*x) * 0.5f + (x*x*x) / 6.0f;
}

float host_custom_fabs(float x) {
    return x < 0.0f ? -x : x;
}

void host_matmul_score(float *q, float *k, float *score, int len, int dim) {
    float scale = 1.0f / host_custom_sqrt((float)dim);
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            float s = 0.0f;
            for (int d = 0; d < dim; d++) {
                s += q[i * dim + d] * k[j * dim + d];
            }
            score[i * len + j] = s * scale;
        }
    }
}

void host_softmax(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float maxval = mat[i * cols];
        for (int j = 1; j < cols; j++) {
            if (mat[i * cols + j] > maxval)
                maxval = mat[i * cols + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = host_custom_exp(mat[i * cols + j] - maxval);
            sum += mat[i * cols + j];
        }
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] /= sum;
        }
    }
}

// token 별 결과를 합친 것 (i = row, j = token)
void host_attention_output(float* score, float* v, float* out, int len, int dim) {
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

// 각 head(tasklet)에서 계산한 결과 -> 이걸 합쳐야지 mha의 최종 결과가 되는것 
void concat_heads(const mha_results_t* host_results, float* concat_out) {
    for (int h = 0; h < NUM_HEADS; h++) {
        int head_size = SEQ_LEN * HEAD_DIM;
        float* target = concat_out + h * head_size;
        memcpy(target, host_results->heads[h].out, sizeof(float) * head_size);
    }
}

void compare_results(mha_results_t *host, mha_results_t *dpu) {
    bool equal = true;

    for (int h = 0; h < NUM_HEADS; h++) {
        printf("Head %d Host: ", h);
        for (int i = 0; i < 5 && i < SEQ_LEN*HEAD_DIM; i++)
            printf("%.6f ", host->heads[h].out[i]);
        printf("\n");

        printf("Head %d DPU : ", h);
        for (int i = 0; i < 5 && i < SEQ_LEN*HEAD_DIM; i++)
            printf("%.6f ", dpu->heads[h].out[i]);
        printf("\n");
    }

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {
            float diff = host_custom_fabs(host->heads[h].out[i] - dpu->heads[h].out[i]);
            if (diff > 1e-3f) {
                // printf("[Mismatch] head %d index %d: Host=%f, DPU=%f, diff=%.6f\n",
                //        h, i, host->heads[h].out[i], dpu->heads[h].out[i], diff);
                equal = false;
                // goto done_compare; // 중간에 mismatch 발생하면, 빠져나오도록 
            }
        }
    }

    unsigned int total_cycles = 0;
    printf("\n--- DPU cycles per head ---\n");
    for (int h = 0; h < NUM_HEADS; h++) {
        unsigned int c = dpu->heads[h].cycles;
        printf("Head %d cycles: %u (%.3f ms)\n", h, c, c / 266000.0);
        total_cycles += c;
    }
    printf("Total DPU cycles: %u (%.3f ms)\n\n", total_cycles, total_cycles / 266000.0);

    if (equal) {
        printf("Host == DPU \n");
    } else {
        printf("Host != DPU \n");
    }
}

int main() {
    struct dpu_set_t set, dpu;
    uint32_t nr_dpus;

    DPU_ASSERT(dpu_alloc(1, NULL, &set)); 
    // DPU_ASSERT(dpu_alloc(1, "backend=simulator", &set)); // sim option으로 주는 부분 

    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL)); // 바이너리를 올려줌 
    DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));

    init_input_data(input_Q, SEQ_LEN * HEAD_DIM * NUM_HEADS, 1);
    init_input_data(input_K, SEQ_LEN * HEAD_DIM * NUM_HEADS, 2);
    init_input_data(input_V, SEQ_LEN * HEAD_DIM * NUM_HEADS, 3);

    size_t matrix_size = SEQ_LEN * HEAD_DIM * sizeof(float);

    for (int h = 0; h < NUM_HEADS; h++) { // input_Q + h * SEQ_LEN * HEAD_DIM 이게 host buffer를 의미함
        DPU_ASSERT(dpu_copy_to(set, "DPU_Q", h * matrix_size, input_Q + h * SEQ_LEN * HEAD_DIM, matrix_size));
        DPU_ASSERT(dpu_copy_to(set, "DPU_K", h * matrix_size, input_K + h * SEQ_LEN * HEAD_DIM, matrix_size));
        DPU_ASSERT(dpu_copy_to(set, "DPU_V", h * matrix_size, input_V + h * SEQ_LEN * HEAD_DIM, matrix_size));
    }
    
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    DPU_ASSERT(dpu_prepare_xfer(set, &dpu_results));

    // DPU의 계산 결과를 CPU로 가져오는것 
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, sizeof(mha_results_t), DPU_XFER_DEFAULT));

    // 비교를 위한 CPU host에서의 동일한 계산 
    struct timespec host_t0, host_t1;
    clock_gettime(CLOCK_MONOTONIC, &host_t0);

    for (int h = 0; h < NUM_HEADS; h++) {
        float* q = input_Q + h * SEQ_LEN * HEAD_DIM;
        float* k = input_K + h * SEQ_LEN * HEAD_DIM;
        float* v = input_V + h * SEQ_LEN * HEAD_DIM;
        
        float score[SEQ_LEN * SEQ_LEN];
        float out[SEQ_LEN * HEAD_DIM];

        host_matmul_score(q, k, score, SEQ_LEN, HEAD_DIM);
        host_softmax(score, SEQ_LEN, SEQ_LEN);
        host_attention_output(score, v, out, SEQ_LEN, HEAD_DIM);

        memcpy(host_results.heads[h].out, out, sizeof(out));
    }

    clock_gettime(CLOCK_MONOTONIC, &host_t1);
    double host_elapsed_ms = (host_t1.tv_sec - host_t0.tv_sec) * 1000.0 +
                             (host_t1.tv_nsec - host_t0.tv_nsec) / 1e6;
    printf("\nHost total time: %.3f ms\n", host_elapsed_ms);

    // MRAM/offset/size 출력하는 구문 추가 
    for(int h=0;h<NUM_HEADS;h++){
        printf("Host: head %d MRAM offset: %zu bytes, expected size: %zu bytes\n",
               h, h*matrix_size, matrix_size);
    }

    // head들의 결과를 합치는 과정 추가한 것 (mha니까) -> 
    // 우선 디버깅 끝나면 출력을 얘로만 + compare_results 값을 출력하도록 수정하기 
    float host_concat_out[SEQ_LEN * EMBED_DIM];
    float dpu_concat_out[SEQ_LEN * EMBED_DIM];

    concat_heads(&host_results, host_concat_out);
    concat_heads(&dpu_results, dpu_concat_out);

    compare_results(&host_results, &dpu_results);

    DPU_ASSERT(dpu_free(set));
    return 0;
}
