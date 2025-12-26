#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

GBS=8
MBS=1
TP=2
PP=2
CP=1
EP=2
ETP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --num-experts 4 \
    --moe-permutation-async-comm \
    --moe-grouped-gemm \
    --gemm-gradient-accumulation-fusion \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 0.02 \
    --moe-zero-memory level0 \
    --moe-fb-overlap \
"
RECOMPUTE_ARGS="
    --recompute-activation-function \
    --recompute-activation-function-num-layers 2 \
    --recompute-norm \
    --recompute-norm-num-layers 1 \
    --enable-recompute-layers-per-pp-rank \
"

GPT_ARGS="
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --schedules-method dualpipev \
    --use-distributed-optimizer \
    --untie-embeddings-and-output-weights \
    --reuse-fp32-param \
    --optimizer-selection fused_torch_adamw \
    --num-layers 4 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 1000 \
    --lr 2.0e-4 \
    --lr-decay-style linear \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --disable-bias-linear \
    --no-gradient-accumulation-fusion \
    --position-embedding-type rope \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16 \
    --tokenizer-type Llama2Tokenizer  \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-size 126464 \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $RECOMPUTE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --exit-interval 100

set +x