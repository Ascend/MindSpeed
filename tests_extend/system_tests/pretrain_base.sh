#!/bin/bash

# entry file:  pretrain_gpt_usecase.py
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   VALUE="${ARGUMENT#*${KEY}=}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done

if [[ -z $EXTRA_ARGS ]]; then
  echo "no additional params"
else
  ADDITIONAL_PARAMS=$EXTRA_ARGS ;
fi
echo "---------------------------------"
set -exo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=TRUE

DEVICES_PER_NODE=${NPU_PER_NODE:-1}
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
    --pipeline-model-parallel-size ${PP_SIZE:-1} \
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
    --train-iters 10 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
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
    --bf16
"

DATA_ARGS="
    --load ${LOAD_CKPT_DIR:-./ckpt_llama} \
    --data-path ${DATA_PATH:-/home/dataset/llama2/alpaca_text_document} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 50 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --exit-on-missing-checkpoint \
    ${ADDITIONAL_PARAMS:+$ADDITIONAL_PARAMS} \

set +x
