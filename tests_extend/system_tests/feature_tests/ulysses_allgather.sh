#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

# Change for multinode config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

VOCAB_FILE=/home/dataset/gpt/gpt2-vocab.json
MERGE_FILE=/home/dataset/gpt/gpt2-merges.txt
DATA_PATH=/home/dataset/gpt/processed/my-gpt2_text_document

TP=2
PP=1
CP=4
SEQ_LEN=$((32*1024))
NUM_LAYERS=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-ascend-mc2 \
    --reuse-fp32-param \
    --use-fused-rotary-pos-emb \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --train-iters 30 \
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
    --use-cp-send-recv-overlap \
    --position-embedding-type rope \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --group-query-attention \
    --num-query-groups 2 \
    --attention-mask-type general \
    --reset-position-ids \
    --reset-attention-mask \
    --use-ulysses-allgather-kv \
    --context-parallel-kv-cache-policy full \
    --context-parallel-cache-interval 1 \
    --bf16 \
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
    --no-save-optim \
    --no-save-rng

set +x

