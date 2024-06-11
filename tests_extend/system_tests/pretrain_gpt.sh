#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

NPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./ckpt_gpt
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt
DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 4 \
    --hidden-size 1024 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --kv-channels 64 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --optimizer adam \
    --loss-scale 1 \
    --lr 0.00015 \
    --train-iters 10 \
    --lr-decay-iters 100 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion \
    --sgd-momentum 0.9 \
    --fp16-lm-cross-entropy \
    --apply-query-key-layer-scaling \
    --accumulate-allreduce-grads-in-fp32 \
    --empty-unused-memory-level 1 \
    --openai-gelu \
    --apply-layernorm-1p \
    --apply-residual-connection-post-layernorm \
    --position-embedding-type rope \
    --rotary-seq-len-interpolation-factor 1 \
    --group-query-attention \
    --num-query-groups 1 \
    --dataloader-type cyclic \
    --standalone-embedding-stage \
    --use-cpu-initialization \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --vocab-size 50257 \
    --num-workers 4 \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 10 \
    --eval-iters 10 \
    --log-num-zeros-in-grad \
    --log-params-norm \
    --log-throughput \
    --timing-log-option max \
    --no-barrier-with-level-1-timing \
    --timing-log-level 0 \
    --tensorboard-dir ./tensorboard \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 1000 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --no-log-learnig-rate-to-tensorboard \
    --no-log-loss-scale-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-world-size-to-tensorboard
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
    --no-save-rng \
