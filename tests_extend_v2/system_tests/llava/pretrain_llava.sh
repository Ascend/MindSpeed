#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=8
GBS=$(($WORLD_SIZE*$MBS/$CP))

DISTRIBUTED_ARGS="
  --nproc_per_node $NPUS_PER_NODE \
	--nnodes $NNODES \
	--node_rank $NODE_RANK \
	--master_addr $MASTER_ADDR \
	--master_port $MASTER_PORT
"

GPT_ARGS="
  --num-layers 12 \
	--hidden-size 624 \
	--attention-dropout 0.0 \
	--hidden-dropout 0.0 \
	--num-attention-heads 12 \
	--micro-batch-size 4 \
	--global-batch-size 32 \
	--seq-length 1024 \
	--max-position-embeddings 1024 \
	--train-iters 2000 \
	--lr-decay-iters 320000 \
	--tokenizer-type NullTokenizer \
	--vocab-size 8192 \
	--lr 5e-7 \
	--lr-decay-style cosine \
	--weight-decay 1e-2 \
	--clip-grad 1.0 \
	--lr-warmup-fraction .01 \
	--tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--attention-softmax-in-fp32 \
	--no-gradient-accumulation-fusion \
	--bf16 \
	--no-load-optim \
	--transformer-impl local \
"

DATA_ARGS="
  --img-h 336 \
	--img-w 336 \
	--patch-dim 14 \
	--mock-data \
	--split 949,50,1 \
"

OUTPUT_ARGS="
    --ckpt-format torch \
    --log-interval 1 \
  	--save-interval 2 \
  	--eval-interval 1000 \
  	--eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_vlm.py \
    $GPT_ARGS \
  	$DATA_ARGS \
	  $OUTPUT_ARGS \
	  --distributed-backend nccl