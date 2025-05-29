#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6005
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_PATH="/home/dataset/model/llama-2-7b-hf"

TP=2
PP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --unaligned-linear \
    --transformer-impl local \
    --use-mcore-models \
    --variable-seq-lengths \
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-cpu-initialization \
    --num-layers 8 \
    --hidden-size 3200 \
    --num-attention-heads 25 \
    --group-query-attention \
    --num-query-groups 5 \
    --no-load-optim \
    --no-load-rng \
    --seq-length 1026 \
    --max-position-embeddings 1026 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 1000 \
    --use-rotary-position-embeddings \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --swiglu \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --eval-iters 0 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --use-flash-attn \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --bf16 \
"
DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path "${TOKENIZER_PATH}" \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --ckpt-format torch \
    --no-save-optim \
    --no-save-rng \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
