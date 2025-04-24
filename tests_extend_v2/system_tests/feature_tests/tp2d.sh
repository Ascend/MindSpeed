#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6004
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

DISTRIBUTED_ARGS=(
    --nproc_per_node $NPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
)

MODEL_ARGS=(
    --use-mcore-models
    --no-rope-fusion
    --transformer-impl local
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 8192
    --num-layers 4
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.1
    --hidden-dropout 0.1
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --optimizer-selection fused_torch_adamw
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --no-gradient-accumulation-fusion
    --tokenizer-type Llama2Tokenizer
    --make-vocab-size-divisible-by 1
    --attention-softmax-in-fp32
    --adam-beta1 0.9
    --initial-loss-scale 4096.0
    --adam-beta2 0.95
    --adam-eps 1e-5
    --group-query-attention
    --num-query-groups 8
)

FEATURE_ARGS=(
    --tp-2d
    --tp-x 2
    --tp-y 2
    --enable-overlap-ag-with-matmul
    --enable-overlap-matmul-with-rs
    --enable-backward-overlap-ag-with-matmul
)

MOE_ARGS=(
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-model ${TOKENIZER_MODEL}
    --split 100,0,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4
    --lr 1.0e-6
    --train-iters 1000
    --lr-decay-style cosine
    --min-lr 1.0e-7
    --lr-warmup-fraction 0.01
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

OUTPUT_ARGS=(
    --log-throughput
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 10
)


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${FEATURE_ARGS[@]} \
    ${MOE_ARGS[@]} \

set +x

