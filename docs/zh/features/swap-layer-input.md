# Transformer 层输入卸载

通过卸载 Transformer 层输入，减少训练时保留激活占用的设备内存。

## 使用方法

**后端限制：** 与 `--moe-fb-overlap` 组合时，必须设置 `--transformer-impl transformer_engine`。

在训练配置中添加 `--swap-layer-input`，默认关闭。

`--swap-layer-input` 与 [swap-attention](swap_attention.md) 是独立开关，不需要通过 `--swap-modules` 指定层输入。
