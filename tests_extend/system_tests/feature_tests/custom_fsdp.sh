#!/bin/bash

source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

SEQ_LENGTH=16384
TRAIN_ITERS=2000
MBS=1
GBS=16
TP=1
PP=1
EP=1
CP=1
ROUTER_BALANCING_TYPE='aux_loss'

TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"
DATA_PATH="/home/dataset/llama2/alpaca_text_document"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 0.001 \
    --moe-permutation-async-comm \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --transformer-impl local \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
"

GPT_ARGS="
    --use-mcore-models \
    --kv-channels 128 \
    --qk-layernorm \
    --tokenizer-type Llama2Tokenizer  \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 1 \
    --hidden-size 1024 \
    --ffn-hidden-size 2048 \
    --num-attention-heads 32 \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
"

FSDP_ARGS="
    --use-custom-fsdp \
    --data-parallel-sharding-strategy optim_grads_params \
    --no-gradient-accumulation-fusion \
"


torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $FSDP_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --exit-interval 50
    --distributed-backend nccl \

set +x
