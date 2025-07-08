# fused_ema_adamw 优化器
## 背景与挑战
多模态领域在模型训练过程中往往会额外生成ema模型用于后续任务，因此需要在训练过程中生成和保存ema模型数据，fused_ema_adamw优化器可以在模型训练过程中额外维护一份ema模型参数，在权重保存时ema模型将自动保存到模型权重文件中。

## 解决思路
在训练过程中，fused_ema_adamw优化器会为模型参数维护一份```ema_params```状态，并在每次优化器迭代过程中更新。ema_params状态更新公式如下：<br>

    ema_params = ema_decay * ema_params + (1 - ema_decay) * model_params

```model_params```为模型参数，```ema_decay```为超参数。其中，```ema_decay```可在训练脚本中使用'--ema-decay 数值'来指定，若脚本中未指定，则默认ema_decay为0.9999。<br>

## 使用场景
主要用于需要保存ema模型用于后续任务的多模态训练场景。<br>

## 使用方法
1.在脚本中添加`--optimizer-selection fused_ema_adamw`，可开启fused_ema_adamw优化器，优化器```ema_params```状态保存功能与ema模型权重保存功能会一同开启。<br>
2.在脚本中添加`--ema-decay 数值`，可指定ema_decay，如未指定，则默认为0.9999。<br>

## 使用影响
1.由于fused_ema_adamw优化器在训练时需要额外维护```ema_params```状态，内存开销会有所增加。<br>
2.权重保存时，优化器的```ema_params```优化器状态将会存储在distrib_optim.pt文件中。<br>
3.权重保存时，ema模型权重数据将会存储在model_optim_rng.pt文件中的```ema_model```字段中。<br>

## 注意事项
1.fused_ema_adamw优化器不支持和参数副本复用特性同时开启，使用本优化器时请勿在训练脚本中添加`--reuse-fp32-param`参数。<br>
2.fused_ema_adamw优化器在训练时需要额外维护一份ema数据，内存开销会有所增加。不同的训练配置内存开销增加幅度不同，使用时请根据实际硬件内存变化情况，适当调整训练脚本或模型结构。
