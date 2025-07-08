# Rotary Position Embedding 融合优化

## 背景与挑战

RoPE (Rotary Positional Embedding，旋转式位置嵌入) 是一种位置编码技术，广泛应用于大型语言模型中，用于有效编码文本序列的位置信息。RoPE结合了绝对位置编码的稳定性与相对位置编码的灵活性，同时具备优秀的长度泛化能力。尽管RoPE已经在诸如LLaMA和GLM等多个前沿模型中得到采纳，但PyTorch框架目前尚未提供专门针对RoPE的实现与优化。因此，模型开发者通常需要通过自定义方式来实现RoPE，而这往往伴随着较高的计算和内存开销。

## 解决方案
为了解决上述问题，我们引入了针对Rotary Embedding的融合优化方案。通过将RoPE操作整合为单一算子，我们显著减少了数据传输次数和临时存储需求，进而优化了模型训练的性能。这一优化由MindSpeed通过调用torch_npu侧接口实现，有效提升了RoPE在模型中的执行效率。

## 使用场景

适用于将Rotary Embedding作为位置编码方案的模型架构。

## 使用方法

* 确保模型配置中已设定以下参数：
`--position-embedding-type  rope`

* 同时，启用RoPE融合算子需设置如下参数：
`--use-fused-rotary-pos-emb`

## 使用效果

通过运用融合优化的RoPE算子，模型训练的性能将得到提升，同时有效降低了内存消耗和计算成本，在LLaMA2-7B模型下，性能提升约1%。

