# 自适应负载均衡分布式FA（Adaptive-CP）

## 序列并行背景

在序列并行（context parallelism）中，长序列数据在序列维度上被切分后较为均匀地分布式加载在多个计算设备上。进行Attention计算时每两个片段的序列之间都需要调用一次FA算子来计算其对应的那一部分输出，并根据原掩码（Original Attention Mask）中的对应部分（Sub Mask）指导其注意力计算。较为特殊的是，Ulysses算法在切分序列维度后通过All To All集合通信将切分维度进行了转置，即计算FA时的数据仍然是全序列数据，被切分的维度是头。因此Ulysses不在本特性覆盖范围内，且与该特性正交，可以同时使用。

在计算Attention时切分序列维度的的分布式FA算法中，较为主流的是Ring Attention：通过生成cp size个轮来遍历执行每一组FA计算，并在每一轮计算的时候通信下一轮计算所需的K和V，从而达到通信被掩盖的效果。Ring Attention中每个设备都会保留其序列的Q，只会进行K和V的组合通信，通过在CP group的每个rank之间形成一个ring来按照一定的顺序发送和接受KV，从而保证所有Q和KV的组合都在指定的cp rank上被执行。

## 现有问题

- Ring attention的调度和通信模式是固定的，即按照环状拓扑的特定方向进行KV组合的收发。可以注意到，无论Mask的稀疏形式如何，Ring Attention的通信与调度模式是不会发生变化的，都会以上述形式切割原Mask并遍历所有可能的计算组合，无论每一个计算组合对应的Mask是否是空；
- 在一些稀疏度较高的Attention Mask下，按照Ring的方式切分得到的每个Sub Mask内也许都有计算任务，但密度很低。这时使用Ring Attention可能造成显著的计算资源浪费；
- 当Sub Mask符合FA算子支持的稀疏模式时，其对应的FA计算可以得到优化，但并不是所有Sub Mask都可以拿到算子层面的稀疏优化，所以可能导致同一轮计算的不同CP rank之间的计算负载不平衡；

针对计算资源浪费和计算负载不平衡的问题，在Causal Attention Mask下已经出现了许多对应的解决方案，如Stripped Ring Attention以及MindSpeed中Ring Attention采用的‘Causal’模式下的优化方式。但是针对泛化Mask并没有自适应的、有效的解决方案。

**本特性旨在解决对任意Mask进行分布式FA计算时可能导致的计算负载不平衡问题，在限制通信上限保证计算通信掩盖的同时得到执行轮数最小的调度方案，从而大幅提高分布式FA计算的计算资源利用率并显著提升Attention的性能。**

## 解决方案

Adaptive-CP通过自适应的流程先后完成<u>序列重映射寻优</u>与<u>调度寻优</u>，来达到上述目的。该特性先将稀疏的Attention Mask进行自适应聚合成为局部稠密的版本减少分布式FA中有效计算组合的数量，然后将有效计算组合灵活地分配给特定的cp rank在特定的round进行计算。两个过程共同得到总轮数最小的序列重映射与通信方式。

### 序列重映射：

已知Attention Mask表达的是不同token之间的注意力依赖，其形状与序列顺序强相关。即对于稀疏的Attention Mask，可能存在一种对token进行重排的方式可以将有依赖的token pair聚集在一起，在表达的依赖关系不变的同时形成一个局部高聚集、整体高稀疏的Attention Mask。

Adaptive-CP实现自适应序列重映射的流程为：

1. 对shape为 $[S, S]$ 的原Attention Mask进行降采样（coarsen），得到更粗粒度的shape为 $[S', S']$ 的coarsed mask，默认 $S'=1024$；

2. 将coarsed mask看作样本量为 $S'$、数据特征维度为 $S'$ 的数据并对其进行PCA降维，保留 $K$ 个数据分布范围最广且方差最大的维度，得到shape为 $[S', K]$ 的reduced mask数据，默认 $K=10$；

3. 对reduced mask进行多次聚类数量不同的k-means无监督聚类并选取最合适的聚类结果，给定的聚类数量在 $[n_{min}, n_{max}]$ 范围内取值，旨在将依赖关系最相似的tokens识别出来。

   其中评价聚类结果的方式为：将属于相同cluster的token排布在一起形成长度为 $S'$ 的新序列，根据新序列与cp size将coarsed mask进行重排与切分得到shape为 $[cp\_size, cp\_size]$ 的grid mask。假设 grid mask中有依赖的token pair（计算任务）的数量为 $c_t$、第 $i$ 个cp rank对应的行和列上的计算任务数量为 $c_i$，计算得到 $D_{den}=c_t/(cp\_size * cp\_size)$，$D_{dev} = Deviance(c_i), \quad i \in \{0,1,...,cp\_size\}$，优先选取 $D_{den}$ 最小、如果 $D_{den}$ 并列最小则选择 $D_{dev}$ 最小的聚类结果；

4. 根据选取的聚类结果组合和拼接tokens得到新序列 $S_{opt}$；

如果这个过程不能得到比原序列更好的新序列，则使用原序列；

### 计算任务调度优化：

该调度方案寻优的灵活性体现在，除了KV的通信外还允许进行Q的通信，即可以通过更复杂的通信机制来使能更灵活的调度机制，实现更少计算轮数的分布式FA调度方案。但该灵活性也大幅提升了对通信掩盖的挑战，所以需要对每一轮中的每一个cp rank的通信单位进行限制。

1. 通过遍历给每个round的每个cp rank填入指定的计算任务；
2. 优先遍历并选择使用当前cp rank存储的序列对应的KV的计算任务，再遍历使用当前cp rank存储的序列对应的QO的计算任务。注意每个cp rank不会被分配到KV或QO对应的序列都在其他cp rank上的计算任务，因为对应的通信量太大；
3. 在遍历过程中保持对通信单位的限制，如果遍历到的分配方式会导致一个cp rank的一个round的通信单位超过上限，则会继续遍历其他位置分配该计算任务；

## 使用方法

使用Adaptive-CP必须保存全序列的Attention Mask，手动调用以下接口在合适的位置根据需要设置attention mask：

```python
from mindspeed.model.transformer import set_attention_mask

set_attention_mask(attn_mask)
```



在启动脚本中根据需要设置以下arguments使能和配置Adaptive-CP：

- 必须配置的变量

```shell
# 使能Adaptive-CP，注意Adaptive-CP和Ring-CP是二选一的关系；
--context-parallel-algo adaptive_cp_algo \
# 使能Hybrid Adaptive-CP，Ulysses和Adaptive-CP同时生效
--context-parallel-algo hybrid_adaptive_cp_algo \
# 必须设置成general类型的attention mask
--attention-mask-type general \
```

当总序列的attention mask占用的内存不大（小于等于64K）且attention mask不发生变化时，只需要设置上面的两个变量；



- 选择性配置的变量

```shell
# 如果配置的mask被放在CPU上，需要开启这个配置
--attention-mask-on-cpu \
# 如果attention mask在每个batch中不一样，则需要开启动态Adaptive-CP（attention mask在不同head中不一样等其他动态场景不支持）
--adaptive-cp-dynamic-attn-mask \
# 如果original attention mask已经是局部聚合的稀疏形式，则可以选择不做序列重映射，使寻优过程产生的时间消耗更少
--adaptive-cp-only-reschedule
# 如果原attention mask的shape不是1024的倍数，或希望通过增加寻优耗时来获得更优的寻优结果，可以开启不做数据降采样
--adaptive-cp-without-coarse \
# 如果想要手动配置mask list来进一步减少寻优过程中H2D和选取mask list产生的时间消耗，可以修改generate_adaptive_cp_mask_list_by_user函数并开启此配置
--adaptive-cp-manually-set-mask-list \
```

配置建议：

1. Adaptive-CP默认开启rescale融合算子，即不论是否开启--use-fused-ring-attention-update都会使能fused ring attention update；

2. 如果attention mask不发生变化，则不需要开启 --adaptive-cp-dynamic-attn-mask，此时只会在第一个step的第一个micro batch开始的时候进行一次寻优。此时可以根据需要进行任意配置，因为寻优耗时再长也只进行一次；

3. 当attention mask随着batch发生变化（动态attention mask场景）时，开启 --adaptive-cp-dynamic-attn-mask，但每个micro batch都会产生一定的寻优耗时。可以根据以下描述选择相关配置降低寻优耗时；

4. 当attention mask虽然整体稀疏但是局部聚合时，比如SWA或document-mask，可以不进行重排寻优只进行调度寻优，打开 --adaptive-cp-only-reschedule 来显著减少寻优耗时；

5. 如果需要做重排和调度，则优先将attention mask放在CPU上，即开启 --attention-mask-on-cpu，寻优过程与前反向执行过程并行，只要前反向执行用时较长就可以将寻优的耗时完全掩盖；

6. 如果只需要做调度（不需要做重排），且全attention mask不会给NPU造成内存压力（总序列长度小于等于64K），建议将全attention mask放在NPU上，即不开启 --attention-mask-on-cpu，从而获得更快的寻优速度；

7. 当开启了--attention-mask-on-cpu，则需要同时传入CPU tensor作为attention mask。这个方法可以减小NPU的内存压力，但会使得寻优过程耗时显著增加（计算得到新序列映射的mask list的过程在CPU上执行效率较低）。此寻优过程在CPU上会与NPU的前反向计算并行执行，如果step用时较长（超过10s）则不需要考虑这部分寻优过程耗时的增加；

8. 当总序列长度不是1024的倍数，不能使用默认的降采样，需要开启 --adaptive-cp-without-coarse，但如果序列长度较大可能引入非常显著的寻优耗时。此配置不建议在动态attention mask场景下开启；

9. 当attention mask是动态的但存在一定的规律且打开 --adaptive-cp-only-reschedule 时，建议打开 --adaptive-cp-manually-set-mask-list 开关，然后修改mindspeed/core/context_parallel/utils.py中的generate_adaptive_cp_mask_list_by_user和get_adaptive_cp_grid_mask_by_user函数，需要自行实现根据输入的新序列和调度信息来生成需要的mask list的代码。

   该函数的两个入参分别为opt_seq和scheduling_info：其中opt_seq是一个长度为总序列长度的list，代表序列重映射寻优过程得到的新序列的index list；其中scheduling info是一个list of list，scheduling_info[i][j]代表第i轮第j个cp rank的计算任务的index $t$，可以根据 t / cp_size 和 t % cp_size 来分别获得该计算任务对应的q和kv的序列片段序号，序列片段序号 $p$ 代表opt_seq被均匀切分成cp_size份之后的第 $p$ 份序列片段。使用该方法需要用户自行保证该函数的正确性；

## 预期效果

在全序列attention mask为full mask或下三角mask时，不要使用Adaptive-CP。在泛化的Attention Mask场景下，性能提升幅度根据attention mask的形状与稀疏度而有所不同，step端到端性能平均提升10-20%。

## 局限说明

- 不支持所有动态场景：当前只支持每个batch的attention mask不同的场景，对于每个head/每个layer/每个step的attention mask不同的动态场景不支持；
- 在自适应序列重映射的算法中，对于横竖向稀疏的mask有更好的效果，斜向稀疏或随机稀疏的attention mask效果较差；
- 在Adaptive CP size较小时，往往不能获得显著的性能提升，因为切分得越细才能得到越多全空的sub mask并省去对应部分的计算任务；
- 当通信带宽较小时可以对应地减小每次计算可掩盖的通信单位上限（默认值为6），可以修改mindspeed/core/context_parallel/utils.py中的宏COMM_THRESHOLD的值；
- 当原attention mask密度较高、稀疏度较低时，Adaptive-CP不能获得很好的效果；
