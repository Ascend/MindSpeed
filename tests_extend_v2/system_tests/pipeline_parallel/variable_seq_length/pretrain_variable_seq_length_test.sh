#!/bin/bash
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=TRUE

DEVICES_PER_NODE=${NPU_PER_NODE:-4}
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($DEVICES_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="
    --nproc_per_node $DEVICES_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP_SIZE:-1} \
    --pipeline-model-parallel-size ${PP_SIZE:-4} \
    --transformer-impl local \
    --sequence-parallel \
    --num-layers 12 \
    --hidden-size 512 \
    --ffn-hidden-size 1376 \
    --num-attention-heads 8 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL:-/home/dataset/model/llama-2-7b-hf/tokenizer.model} \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size ${MBS:-2} \
    --global-batch-size ${GBS:-16} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-5 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type learned_absolute \
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-9 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --mock-data \
    --npu-deterministic \
    --variable-seq-lengths \
    --moe-token-dispatcher-type alltoall
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 10 \
    --eval-iters 2 \
"

torchrun $DISTRIBUTED_ARGS pretrain_variable_seq_length_test.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --exit-on-missing-checkpoint \
