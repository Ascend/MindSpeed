#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./ckpt_gpt
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt
DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence

TP=2
PP=1
CP=2
EP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

MOE_ARGS="
    --expert-model-parallel-size ${EP} \
    --moe-model-type deepspeed_moe \
    --num-experts 2 \
    --moe-router-topk 2 \
    --enable-token-rearrange-opt \
    --use-pipe-experts \
    --pipe-experts-multi-stream \
    --pipe-experts-multi-data 4 \
    --ampipe-degree 2 \
    --ampipe-tp-sp-comm-overlap \
    --moe-no-drop \
    --moe-use-sinkhorn \
    --moe-dynamic-padding \
"

GPT_ARGS="
    --use-legacy-models \
    --tensor-model-parallel-size ${TP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-cp-send-recv-overlap \
    --reuse-fp32-param \
    --sequence-parallel \
    --use-fused-rotary-pos-emb \
    --use-flash-attn \
    --use-distributed-optimizer \
    --num-layers 2 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --train-iters 1000 \
    --lr-decay-iters 320000 \
    --lr 5.0e-7 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --no-gradient-accumulation-fusion \
    --position-embedding-type rope \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --vocab-size 50257 \
    --num-workers 4 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
    --log-throughput \
    --timing-log-option max \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $RECOMPUTE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \

set +x
