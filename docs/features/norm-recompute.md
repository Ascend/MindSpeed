# Norm重计算

## 问题分析

大模型训练过程中，往往会面临的显存不足的问题。

## 解决方案

类似于激活函数重计算，本特性支持了Norm层的重计算。

## 解决思路

运用激活函数重计算特性中的 `checkpoint` 机制，对norm层进行重计算处理，具体细节如下文所示：
[原文链接](https://www.usenix.org/conference/atc24/presentation/yuan)

## 使用场景

主要用于训练场景，用户内存不足或要进一步节省内存时。

## 使用方法

脚本中添加：`--recompute-norm` 可开启Norm重计算。此特性仅支持mcore分支。

添加：`--recompute-norm-num-layers ${num}` 可指定Norm重计算的层数。

## 注意事项
1. Norm重计算特性仅支持mcore分支，不支持legacy分支，即仅支持在开启`--use-mcore-models`时，通过`--recompute-norm`使能。
2. Norm重计算兼容激活函数重计算、全重计算同时开启：
   - 同时开启时，仅支持 `--recompute-method` 为 `block`。
   - 同时开启时，会按照指定的全重计算和Norm重计算的层数做各自类型的重计算，即不会有一层既做全重计算又做Norm重计算。
   - 同时开启时，执行优先级是先计算全重计算层，后计算Norm重计算层。
