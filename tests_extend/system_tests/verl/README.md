# VERL System Tests

该目录汇总 MindSpeed 对 VERL 的系统级测试脚本，每个模型一个子目录。

## 目录结构

- `qwen3_30b/`：Qwen3-30B-A3B 训练脚本与说明
- `qwen3_8b/`：Qwen3-8B 训练脚本与说明

## 使用方式

1. 进入对应模型目录查看 `README.md`。
2. 按说明准备权重、数据与环境。
3. 在 `verl` 代码仓内执行脚本。

```bash
cd verl
```

## 新增脚本约定

- 每个模型一个子目录。
- 每个模型目录需包含 `README.md`，以及至少一个可执行脚本。
