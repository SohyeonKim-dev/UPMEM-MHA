#ifndef __COMMON_H__
#define __COMMON_H__

#define BATCH_SIZE 1 // 한번에 MHA로 처리할 샘플 수 : 지금은 1로 잡음 (하나만 처리)
#define EMBED_DIM 128 // embedding vector의 dimension : 보통은 다 실수 값이 들어감 -> 양자화 가정하고 구현하기
#define SEQ_LEN 32 
#define NUM_HEADS 1
#define HEAD_DIM 16
// #define HEAD_DIM (EMBED_DIM / NUM_HEADS)

#ifndef NR_TASKLETS
#define NR_TASKLETS NUM_HEADS // head와 TASKLETS 동일하도록 (== 2)
#endif

// NR_TASKLETS도 default 16 (헤드 별 병렬처리 -> 동일하게 사용) -> 지금은 2개 뿐 
// It is recommended to implement algorithms with 16 active tasklets to absorb the latency of memory accesses.
// https://sdk.upmem.com/stable/fff_CodingTips.html#data-sharing-label 

#include <stdint.h>

typedef struct {
    float out[SEQ_LEN * HEAD_DIM]; // mha output 저장
    uint32_t cycles;                
} mha_result_t;

// 각 tasklet의 mha_result_t 모으기 (head수만큼)
typedef struct {
    mha_result_t heads[NUM_HEADS];
} mha_results_t;

#endif
