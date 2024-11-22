source "tests_extend/system_tests/env_npu.sh"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6900
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model load ckpt path"
DATA_PATH="your data path"
SCRIPT_CONFIG="tests_extend/system_tests/opensora1.0/16x256x256.py"

TP=2
PP=2
CP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-legacy-models \
    --tensor-model-parallel-size ${TP} \
    --use-multiparameter-pipeline-model-parallel \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size 4 \
    --global-batch-size 4 \
    --num-layers 28 \
    --hidden-size 1152 \
    --num-attention-heads 16 \
    --seq-length 1024\
    --max-position-embeddings 1024 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --train-iters 2500 \
    --weight-decay 0 \
    --weight-decay 0.0 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --use-ema
"

DATA_ARGS="
    --data-path $DATA_PATH \
"

MODEL_ARGS="
    --additional-config $SCRIPT_CONFIG \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"
torchrun $DISTRIBUTED_ARGS pretrain_opensora.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MODEL_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save ${CKPT_SAVE_DIR} \

set +x
