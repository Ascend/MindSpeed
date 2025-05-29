#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP=2
PP=2
CP=2
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1 \
    --enable-recompute-layers-per-pp-rank \
    --recompute-activation-function \
    --recompute-activation-function-num-layers 1 \
    --recompute-in-advance \
"

GPT_ARGS="
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --use-ascend-mc2 \
    --reuse-fp32-param \
    --sequence-parallel \
    --use-fused-rotary-pos-emb \
    --use-fused-rmsnorm \
    --use-flash-attn \
    --num-layers 4 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --lr-warmup-fraction 0.01 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 50 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $RECOMPUTE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --exit-interval 50 \
    --save $CKPT_DIR \
    --ckpt-format torch \

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $RECOMPUTE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load $CKPT_DIR \
    --ckpt-format torch \

set +x
