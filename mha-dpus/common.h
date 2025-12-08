#ifndef __MHA_COMMON_H__
#define __MHA_COMMON_H__

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define BATCH_SIZE 128

#define EMBED_DIM 512
#define SEQ_LEN 128
#define HEAD_DIM 16
#define NUM_HEADS 16   

#define TOTAL_SLOTS (NUM_HEADS * BATCH_SIZE)
#define SLOTS_PER_DPU 1

#define NR_DPUS TOTAL_SLOTS

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#define QK_SCALE 127
#define V_SCALE 127

typedef struct {
    int32_t out[SEQ_LEN * HEAD_DIM];  
    uint64_t cycles;
} mha_result_t;

typedef struct {
    mha_result_t heads[TOTAL_SLOTS];
} mha_results_t;

#endif 
