# 支持EOD Reset训练场景

## EOD Reset训练场景
通常一个批次中输入进模型的文本序列是由多个文档（doc）拼接得到。在默认情况下，多个文档被视为同一序列，互相间的self attention没有掩盖。在特定情况下，多个文档间要求独立，文档间不能互相做self attention，在这种情况下attention mask和position ids需要在每个文档结束的位置（EOD）被重新设置。--reset-position-ids参数关闭时，整个序列计算位置编码；开启时，在每个序列内独立计算位置编码。

## 解决方案
通过调用底层flash-attention算子的可变长模式，支持EOD Reset训练场景。同时在EOD Reset训练场景下，支持Ring Attention长序列并行，对超长序列场景进行加速。

## 使用方式
### 1. 数据准备
（1）首先确保每一个文档的末尾都添加了EOD Token  
（2）对于--attention-mask-type为causal的情况，需要保证每个子序列都被pad到了2*cp的长度

### 2. 参数设置
（1）打开`--reset-attention-mask`选项  
（2）使用`--reset-position-ids`选项，来代表位置编码是否reset  
（3）--attention-mask-type可以指定为causal或者general，两者计算结果等价。causal为加速实现，general为基线方案

### 3. 注意事项
Ascend EOD Reset训练场景下mask-type为general时，Ring/Hybrid Attention比Ulysses下降较多，为正常现象；
mask-type为causal时，使用加速方案。