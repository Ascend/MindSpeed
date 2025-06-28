# TestOp

#### 介绍
测试算子效率工具


#### 支持算子
allgather_matmul, matmul_reduce_scatter, matmul_all_reduce

#### 支持通算融合并行策略
mc2, coc_kernel, coc

#### 使用说明
##### config.py配置说明
1.data_type:数据类型, 支持bf16和fp16  
2.shape_list:算子输入shape，三轴MKN格式  
3.comm_overlap_type:通算融合并行策略, 支持mc2, coc_kernel, coc，使用枚举类型  
4.model_ops_shape：整网中使用的融合算子及shape，用于计算整网最优通算融合选择
##### 前置依赖
1.安装本项目requirements.txt中的相关依赖包  
2.安装matplotlib、openpyxl包

##### 运行说明

1.安裝CANN包、CANN-NNAL(如使用ATB算子)和torch_npu等必要的依赖包
2.参考Readme安装MindSpeed仓库
3.在Megatron-LM路径下执行 bash tests_extend/tools/coc_tools/start.sh

##### 运行结果说明
1.根据data_type、shape_list和comm_overlap_type计算各策略各算子执行时长，保存为折线图和表格，名称为plot.png和matrix_multiplication_times.xlsx  
2.根据data_type、comm_overlap_type和model_ops_shape计算整网最优通算融合选择，以打屏日志的形式输出  

##### 注意事项
1.shape_list的长度为1时，仅有一组数据折线图无法显示折线，请使用表格数据进行查看  