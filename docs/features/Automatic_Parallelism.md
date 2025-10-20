## Automatic Parallelism

## 背景与挑战

当前主流的大模型并行训练方法有PP、TP、DP、SP、CP、Ulysses Parallel（UP）、VPP、EP等，在内存、计算、通信方面都有不同的优化，直接叠加。大模型端到端训练性能由模型结构、集群规模、并行配置、batch_size等因素共同决定，在调优时需要综合考虑。当前并行配置人工调优需要大量的专家经验、人工分析和实验调优，预计数天~数周，实验成本高。相似模型的最优并行配置也并不相同，仍需花费时间进行优化。随着搜索空间变大，依赖手工调优变得不可行。例如，llama65B模型在4*8的集群规模下，仅考虑PP、TP、DP、SP、VPP、mbs六个维度，配置组合有812种，手工调优时间成本太高。因此，需要构建自动并行系统根据模型结构和集群规模给用户自动推荐一个性能较优的并行配置策略。

## 解决方案

针对该问题场景提出多维并行配置自动寻优算法，在给定模型结构、集群配置的条件下，用户仅需要在启动脚本中配置相关参数即可启动多维并行配置自动寻优，在规定时间内找到较优的并行配置推荐给用户。算法原理图如下：

* **内存自适应感知的搜索空间构建**：考虑模型结构和集群信息约束，采用内存灰盒模型排除OOM并行配置，缩小搜索空间；
* **基于算子不确定性估计的高保序性Cost Model建模方法**：引入低保真数据（单算子调用）作为先验信息，结合算子整网性能数据构建算子执行耗时的不确定性模型，结合通信耗时根据并行策略合成得到端到端性能的概率分布模型。
* **基于概率匹配的高效搜索算法**：基于Thompson Sampling方法探索并行策略，以高概率探索高价值并行配置，提高探索效率，灵活支持探索早停，提高易用性。


![1](../../sources/images/auto_parallel_1.png)

**并行配置的支持情况：**

已支持搜索的并行配置维度：

- [x] PP
- [x] TP
- [x] DP
- [x] CP
- [x] DeepSpeed-Ulysses
- [x] Megatron-SP
- [x] mbs

正在支持的并行配置维度：

- [ ] MOE
- [ ] VPP
- [ ] 自适应重计算

## 使用方法

在使用多维自动并行特性时，**需使用python作为脚本启动器，在所有的节点上拉起脚本**，并配置多维自动并行相关的参数。相关参数及其函数如下表所示：

| 参数名           | 参数含义                                          |
| ---------------- | ------------------------------------------------- |
| --auto-parallel  | 多维自动并行特性总开关                            |
| --nnodes          | 集群中节点的个数                                  |
| --nproc-per-node | 每个节点中计算设备的个数                               |
| --master-addr    | 集群中主节点的IP地址                              |
| --master-port    | 用于通信的端口号，各节点需要配置相同的端口号      |
| --node-rank      | 集群中节点的rank，主节点为0，其他节点为1,2,······ |

注：算法最长搜索时间为8小时，支持灵活提前退出，无需人工干预。

下面是基于llama7B模型的配置示例：

```shell
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

MASTER_ADDR=localhost
MASTER_PORT=6001
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

LOAD_CHECKPOINT_PATH=./ckpt
SAVE_CHECKPOINT_PATH=./ckpt
DATA_PATH={your dataset path}
TOKENIZER_MODEL={your tokenizer model path}
TP=1
PP=8

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
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 5000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --load ${LOAD_CHECKPOINT_PATH}  \
    --no-load-optim \
    --no-load-rng \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

SEARCH_ARGS="
    --auto-parallel \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    --node-rank $NODE_RANK \
"

python pretrain_gpt.py \
     $GPT_ARGS \
     $DATA_ARGS \
     $OUTPUT_ARGS \
     $SEARCH_ARGS \
     --distributed-backend nccl \
     | tee logs/search_llama_7b.txt
```

## 使用效果

![2](../../sources/images/auto_parallel_2.png)

