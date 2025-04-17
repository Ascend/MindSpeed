#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6030"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --sequence-parallel
    --use-distributed-optimizer
)

MODEL_ARGS=(
    --use-mcore-models
    --no-rope-fusion
    --npu-deterministic
    --transformer-impl local
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 8
    --hidden-size 1024
    --ffn-hidden-size 4096
    --num-attention-heads 32
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL:-/home/dataset/model/llama-2-7b-hf/tokenizer.model}
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --use-cpu-initialization
    --no-masked-softmax-fusion
    --no-position-embedding
)


FEATURE_ARGS=(
    --use-fused-rmsnorm
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 1
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
)

DATA_ARGS=(
    --data-path ${DATA_PATH:-/home/dataset/llama2/alpaca_text_document}
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1e-4
    --train-iters 3000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 20000 \
    --eval-interval 20000 \
    --eval-iters 10 \
    --no-load-optim \
    --no-load-rng \
    --ckpt-format torch
)


PROFILE_ARGS=(
#    --profile
#    --profile-step-start 10
#    --profile-step-end 12
#    --profile-level level1
#    --profile-with-cpu
#    --profile-with-memory
)


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${FEATURE_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${PROFILE_ARGS[@]} \
