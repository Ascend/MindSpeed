#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

# Change for multinode config
NPUS_PER_NODE=8
MASTER_ADDR=<master_ip_address>
MASTER_PORT=6000
NNODES=8
NODE_RANK=<local_rank>
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./ckpt_gpt
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt
DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence

TP=8
PP=4
CP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --context-parallel-size ${CP} \
    --use-fused-rotary-pos-emb \
    --use-cp-send-recv-overlap \
    --context-parallel-algo megatron_cp_algo \
    --use-ascend-mc2 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 7 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 48 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 5000 \
    --lr-decay-iters 320000 \
    --lr 5.0e-7 \
    --min-lr 5.0e-8 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --no-gradient-accumulation-fusion \
    --use-flash-attn \
    --position-embedding-type rope \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
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
    --no-barrier-with-level-1-timing \
    --timing-log-level 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --distributed-timeout-minutes 10 \
    --seed 1234 \
    --save $CHECKPOINT_PATH \
    --no-save-optim \
    --no-save-rng

set +x
