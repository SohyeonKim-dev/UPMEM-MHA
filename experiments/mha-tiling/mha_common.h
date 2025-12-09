#ifndef __MHA_COMMON_H__
#define __MHA_COMMON_H__

#include <stdint.h>
#include <stdbool.h>

#define BATCH_SIZE 1
#define EMBED_DIM 128
#define SEQ_LEN 32
#define NUM_HEADS 16
#define HEAD_DIM 8

#ifndef NR_TASKLETS
#define NR_TASKLETS NUM_HEADS
#endif

#define QK_SCALE 127
#define V_SCALE 127

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int32_t out[SEQ_LEN * HEAD_DIM];   
    uint64_t cycles;
} mha_result_t;

typedef struct {
    mha_result_t heads[NUM_HEADS];
} mha_results_t;

#endif
