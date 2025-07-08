# Context Parallelism特性中的KV缓存优化

## 背景与挑战
Context Parallelism特性在attention计算前将序列在sequence维度切分，在attention计算时，需要计算全量的序列，因此在attention计算时，会有CP的通信产生。

1. Ring CP在attention计算时通过send/recv方式循环接收其他卡的KV，最终保证Q能关注到全量KV，保持和不拆解情况的数学等价性。当前，前向计算完成后KV被丢弃，反向时需要再次send-recv拿到KV。 当在短序列计算过程中通信时间会大于计算时间，导致计算时间无法掩盖通信时间的情况，从而导致性能变差。因此，需要针对计算时间不足以掩盖通信时间的情况进行优化来加速该场景下的训练需求。

2. Ulysses CP方案在GQA模型下，开启TP后，每个rank通常会只有一个head，在这种情况下，使用All2All的通信量与AllGather通信量相同，而All2All方案在只有一个head的情况下，需要对KV进行repeat，在数据layerout通常为sbh或sbnd的情况下，对h维做repeat，地址不连续，会导致算子存在效率问题，并且需要插入transpose等操作，而AllGather直接操作s维，地址连续，无需额外操作。

3. Ulysses CP在有repeat产生的情况下，传入attention反向的Key和Value相较于repeat前的Key和Value内存扩大了CP倍，这将会导致内存的消耗增加出现out of memory的情况。

## 解决方案

1. 在Ring Attention长序列并行的基础上加入KV缓存功能，可选择进行(1)缓存所有K,V，(2)只缓存K以及(3)设置分层缓存的方式在长序列并行的前向中使前向计算接收的KV始终保留至反向计算，直接计算出梯度结果，减少通信时间。

2. 在GQA模型，一个head的情况下，Ulysses Attention长序列并行的基础上加入AllGather KV + All2All Q的方案，减少repeat操作以及transpose等内存非连续的开销，提高训练性能。

3. 在Ulysses使用All2All和AllGather方案加入KV缓存功能，可选择进行(1)缓存所有K,V，(2)只缓存K以及(3)设置分层缓存的方式在前向中将通信前的KV进行缓存始终保留至反向再进行重新通信进行计算，节省内存。All2All方案只能在做了Repeat的情况下可以开启KV缓存。

### 解决思路:
1. Ring方案中序列被切分成CP份并行计算，在不同rank上计算出自己的K和V，同时send-recv其他rank的K和V。例如rank0上的K0/V0和K7V7发送给“下游”的rank，同时接收“上游”rank发送过来的K3/V3和K4/V4，每张卡重复执行相同的动作CP-1次，最终每个切分后的序列可以“关注”到全局的KV，计算得到完整attention结果。反向计算逻辑同理，初始时每个rank有自己的KV，在计算出自己的gradient后，之后步骤将接收到的K和V分块以及dK和dV发送给其他rank，同时接收其他rank的K、V分块以及dK和dV分块，并把接收到的K和V作为输入计算和更新梯度，实现计算和通信并行。
反向过程关键的一点，rank间通信需要发送K、V、dK、dV四个数据块，一共要发送CP-1次，其中K和V在前向已经在各个rank间逐次接收发送，如果在前向过程中将K、V缓存，反向的通信时间将减半。在CP比较大时，缓存全部K、V对内存压力增大，通过支持缓存K、V的一部分，或者每经过N个Layer缓存一次，支持按需灵活配置。

2. 在GQA模型，一个head的情况下，使用AllGather KV的通信方式替换原有的Repeat-All2All KV方式获取全量的sequence，对Q仍然使用All2All方案。

3. Ulysses方案中，将在前向进行Repeat-All2All或者AllGather通信前的KV进行缓存带到反向，并使用通信后的KV进行计算确保计算的正确性，反向在拿到Repeat-All2All或者AllGather通信前的KV的时候，对KV进行Repeat-All2All或者AllGather重新通信进行梯度计算。因为进行重新通信会有性能损失，因此可以缓存K、V的一部分，或者每经过N个Layer缓存一次，灵活组合，在内存限制内达到最优的性能。

灵活缓存方案如下，
1. 支持配置缓存K、V的layer间隔：缓存部分K、V可通过考虑在不同layer之间进行缓存来实现，通过增加一个参数interval来控制缓存的间隔层数。例如interval=1时，那么就会在编号为0，2，4，...的layer中对K、V进行缓存，依次类推。缓存间隔支持从0开始，不超过rank上的layer数量，间隔默认值等于0。

2. 支持缓存K、V的一部分：在每个layer上，可支持只缓存K（K和V的size一样），这种方法通过使用一个参数对其控制，当参数的值为half时，只对K缓存，配置full则缓存K和V，默认缓存K和V。此配置和按layer间隔配置缓存可同时开启，配置后的缓存效果叠加，互不冲突

## 使用场景

训练过程中开启长序列并行的情况下。

需使用FlashAttention，目前已默认开启FlashAttention。

在Ring Attention中想要使用KV缓存获得收益，需要使得计算时间小于通信时间，理论上需要确保每个计算块分到的序列长度需要`c < F/B`。其中`F`是每个device的FLOPS，`B`是每个device间的带宽。

在Ulysses Attention中，想要使用AllGather KV + All2All Q获得收益，需要使用GQA模型，并且需要在通信量相同的前提下，即KV仅有一个head的情况下。

在Ulysses Attention中，想要使用KV缓存获得收益，Repeat-All2All方案需要在使用repeat的情况下，才能获得内存收益，而AllGather KV + All2All Q开启CP即可以获得内存收益。

## 使用方法

| 重要参数                                           | 参数说明                                                     |
|------------------------------------------------|----------------------------------------------------------|
| --context-parallel-kv-cache-policy [full/half] | 开启CP前向计算过程缓存KV及其级别，默认full缓存K和V，half缓存K                   |
| --context-parallel-cache-interval [int]        | 设定执行CP前向计算过程缓存KV的layer间隔层数，默认为0，即每一个layer都需要缓存,根据用户需求配置。 |
| --use-ulysses-allgather-kv                     | 设定Ulysses Attention启用AllGather方案，默认为False，不启用。           |

## 使用效果

在Ring Attention中计算时间无法掩盖通信时间的场景下，开启KV缓存特性会使得训练时间变短，提升训练性能，但会导致内存增加。

在Ulysses Attention中开启AllGather KV，在允许的场景下，会使得训练时间变短，提升训练性能。

在Ulysses Attention中开启KV缓存，在Repeat-All2All做了Repeat的情况下，内存使用会减少，但会导致性能下降。 AllGather情况下，内存使用会减少，但会导致性能下降。

## 注意事项：

1. 开启--context-parallel-kv-cache-policy时需要同时开启Context Parallel，否则特性不支持。
2. 开启--context-parallel-cache-interval时需要同时开启--context-parallel-kv-cache-policy并且interval的值需要小于layer的数量，否则特性不支持。
3. 开启--use-ulysses-allgather-kv时需要开启Context Parallel且设置--context-parallel-algo ulysses_cp_algo，并且需要开启--group-query-attention，且KV每个rank的head数量为1， 否则特性不支持。
4. 开启--context-parallel-kv-cache-policy以及--context-parallel-algo ulysses_cp_algo的情况下，需要使KV做Repeat操作，否则特性不支持。