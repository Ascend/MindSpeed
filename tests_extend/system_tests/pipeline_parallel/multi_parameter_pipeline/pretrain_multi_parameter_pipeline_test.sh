# 本测试脚本，只支持以下情况
# (1) world_size=1, 不开启并行特性
# (2) world_size=4, PP=4, 同时打开--use-multiparameter-pipeline-model-parallel开关， 进行PP测试
# (3) world_size=4, PP=4, 同时打开--use-multiparameter-pipeline-model-parallel \
# 与 --num-layers-per-virtual-pipeline-stage 1， 进行VPP测试
# 三者精度应当对齐


export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6900
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
TOKENIZER_MODEL=""

TP=1
PP=4
CP=1
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size 2048 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL:-/home/dataset/model/llama-2-7b-hf/tokenizer.model} \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 100 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --min-lr 1.0e-5 \
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
    --mock-data \
    --npu-deterministic \
    --optimization-level 2 \
    --transformer-impl local \
    --use-multiparameter-pipeline-model-parallel
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_multi_parameter_pipeline_test.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
