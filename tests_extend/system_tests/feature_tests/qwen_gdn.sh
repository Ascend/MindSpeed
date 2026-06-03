#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

########################################
# Distributed Config
########################################

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

CHECKPOINT_PATH=./ckpt_gpt

########################################
# Dataset Config
########################################

DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --vocab-size 50257 \
    --num-workers 4 \
    --split 949,50,1
"

########################################
# GPT Model Config
########################################

TP=1
PP=2
SEQ_LEN=1024
GLOBAL_BATCH_SIZE=64
ITERS_STEP=1000

LINEAR_KEY_HEAD_DIM=128
LINEAR_VALUE_HEAD_DIM=128
LINEAR_NUM_KEY_HEADS=16
LINEAR_NUM_VALUE_HEADS=32
LINEAR_CONV_KERNEL_DIM=4
LINEAR_ATTENTION_FREQ=4

LINEAR_ARGS="
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq ${LINEAR_ATTENTION_FREQ} \
    --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM} \
    --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM} \
    --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM} \
    --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS} \
    --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS}
"

GPT_ARGS="
    --transformer-impl transformer_engine \
    --no-create-attention-mask-in-dataloader \
    --no-gradient-accumulation-fusion \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 64 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size 1 \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters ${ITERS_STEP} \
    --init-method-std 0.01 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096.0 \
    --disable-bias-linear \
    --lr-warmup-fraction 0.01 \
    --bf16 \
    --use-flash-attn \
    --norm-epsilon 1.0e-6 \
    --rotary-base 1000000 \
    $LINEAR_ARGS
"

########################################
# Distributed Launch Config
########################################

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

########################################
# Output / Logging Config
########################################

OUTPUT_ARGS="
    --save ${CHECKPOINT_PATH} \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10
"

########################################
# Run Training
########################################

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl

set +x
