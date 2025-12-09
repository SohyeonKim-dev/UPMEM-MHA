#!/bin/bash

COMMON_H="common.h"
LOGFILE="log.txt"

echo "==== Experiment Log $(date) ====" > $LOGFILE

update_macro() {
    macro=$1
    value=$2
    sed -i "s/#define ${macro} .*/#define ${macro} ${value}/" $COMMON_H
}

compile_dpu() {
    echo "[*] Compiling dpu.c"
    dpu-upmem-dpurte-clang -I/home/coslab/upmem-sdk/include -o dpus.mpo dpu.c >> $LOGFILE 2>&1
}

compile_host() {
    echo "[*] Compiling host.c"
    gcc -O2 -std=c11 -D_POSIX_C_SOURCE=199309L host.c \
        -I/home/coslab/upmem-sdk/include/dpu \
        -L/home/coslab/upmem-sdk/lib \
        -ldpu -lpthread -lm -o host >> $LOGFILE 2>&1
}

run_host() {
    echo "[*] Running ./host"
    echo "---- RUN START ----" >> $LOGFILE
    ./host >> $LOGFILE 2>&1
    echo "---- RUN END ----" >> $LOGFILE
}

SEQ_LIST=(32 48 64 80 96 112 128)

update_macro "BATCH_SIZE" 128
update_macro "EMBED_DIM" 128
update_macro "HEAD_DIM" 16
update_macro "NUM_HEADS" 16

for SEQ in "${SEQ_LIST[@]}"; do
    update_macro "SEQ_LEN" $SEQ

    echo "===== Running SEQ_LEN=$SEQ (BATCH=128) ====="
    echo "[EXP_SEQ] BATCH=128, SEQ_LEN=${SEQ}" >> $LOGFILE

    compile_dpu
    compile_host
    run_host

    echo "" >> $LOGFILE
done

BATCH_LIST=(32 48 64 80 96 112 128)

update_macro "SEQ_LEN" 128

for BATCH in "${BATCH_LIST[@]}"; do
    update_macro "BATCH_SIZE" $BATCH

    echo "===== Running BATCH=$BATCH (SEQ=128) ====="
    echo "[EXP_BATCH] BATCH=${BATCH}, SEQ_LEN=128" >> $LOGFILE

    compile_dpu
    compile_host
    run_host

    echo "" >> $LOGFILE
done

HEAD_DIM_LIST=(16 24 32 40 48 56 64)

update_macro "BATCH_SIZE" 128
update_macro "SEQ_LEN" 32

for HD in "${HEAD_DIM_LIST[@]}"; do
    update_macro "HEAD_DIM" $HD

    echo "===== Running HEAD_DIM=$HD ====="
    echo "[EXP_HD] BATCH=128, SEQ_LEN=32, HEAD_DIM=${HD}" >> $LOGFILE

    compile_dpu
    compile_host
    run_host

    echo "" >> $LOGFILE
done

NUM_HEADS_LIST=(12 16 20 24 28 32)

update_macro "BATCH_SIZE" 64
update_macro "SEQ_LEN" 32

for NH in "${NUM_HEADS_LIST[@]}"; do
    update_macro "NUM_HEADS" $NH

    echo "===== Running NUM_HEADS=$NH ====="
    echo "[EXP_NH] BATCH=64, SEQ_LEN=32, NUM_HEADS=${NH}" >> $LOGFILE

    compile_dpu
    compile_host
    run_host

    echo "" >> $LOGFILE
done

TASKLET_LIST=(4 8 12 16 20 24)

update_macro "BATCH_SIZE" 128
update_macro "SEQ_LEN" 64
update_macro "HEAD_DIM" 32
update_macro "NUM_HEADS" 16

for NT in "${TASKLET_LIST[@]}"; do
    update_macro "NR_TASKLETS" $NT

    echo "===== Running NR_TASKLETS=$NT ====="
    echo "[EXP_TL] BATCH=128, SEQ_LEN=64, HEAD_DIM=32, NUM_HEADS=16, NR_TASKLETS=${NT}" >> $LOGFILE

    compile_dpu
    compile_host
    run_host

    echo "" >> $LOGFILE
done

echo "All experiments finished. Log saved to $LOGFILE"