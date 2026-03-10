# 自定义算子 

 本手册描述自定义算子发布情况，PyTorch框架和MindSpore框架支持情况
 
**表 1**  自定义算子列表
 
|类型|自定义算子名称|Released|PyTorch框架支持情况|MindSpore框架支持情况|
|--|--|--|--|--|
|npu_dropout_add_layer_norm对外接口|[npu_dropout_add_layer_norm](ops/npu_dropout_add_layer_norm.md)|✅|✅|✅|
|npu_rotary_position_embedding对外接口|[npu_rotary_position_embedding](ops/npu_rotary_position_embedding.md)|✅|✅|✅|
|fusion_attention对外接口|[fusion_attention](ops/fusion_attention.md)|✅|✅|✅|
|rms_norm对外接口|[rms_norm](ops/rms_norm.md)|✅|✅|✅|
|swiglu对外接口|[swiglu](ops/swiglu.md)|✅|✅|✅|
|npu_mm_all_reduce_add_rms_norm对外接口|[npu_mm_all_reduce_add_rms_norm](ops/npu_mm_all_reduce_add_rms_norm.md)|✅|✅|✅|
|npu_mm_all_reduce_add_rms_norm_对外接口|[npu_mm_all_reduce_add_rms_norm_](ops/npu_mm_all_reduce_add_rms_norm_.md)|✅|✅|✅|
|gmm对外接口|[npu_gmm](ops/gmm.md)|✅|✅|✅|
|npu_grouped_mat_mul_all_reduce对外接口|[npu_grouped_mat_mul_all_reduce](ops/npu_grouped_mat_mul_all_reduce.md)|✅|✅|✅|
|npu_ring_attention_update对外接口|[npu_ring_attention_update](ops/npu_ring_attention_update.md)|✅|✅|❌|
|npu_matmul_add_fp32对外接口|[npu_matmul_add_fp32](ops/npu_matmul_add.md)|✅|✅|❌|
|npu_groupmatmul_add_fp32对外接口|[npu_groupmatmul_add_fp32](ops/npu_groupmatmul_add.md)|✅|✅|❌|
|npu_apply_fused_ema_adamw对外接口|[npu_apply_fused_ema_adamw](ops/npu_apply_fused_ema_adamw.md)|❌|✅|✅|
|lcal_coc对外接口|[lcal_coc](ops/lcal_coc.md)|❌|✅|✅|
|ffn对外接口|[ffn](ops/ffn.md)|❌|✅|❌|
|npu_alltoall_allgather_bmm对外接口|[npu_all_to_all_all_gather_bmm](ops/npu_all_to_all_all_gather_bmm.md)|❌|✅|❌|
|npu_bmm_reducescatter_alltoall对外接口|[npu_bmm_reduce_scatter_all_to_all](ops/npu_bmm_reduce_scatter_all_to_all.md)|❌|✅|❌|
|quant_gmm对外接口|[quant_gmm](ops/quant_gmm.md)|❌|✅|❌|
|npu_apply_fused_adamw_v2对外接口|[npu_apply_fused_adamw_v2](ops/npu_apply_fused_adamw_v2.md)|✅|✅|❌|

> [!NOTE]
>
> 上表中的“Released”代表商用版本已发布，“✅”代表支持，“❌”代表不支持
