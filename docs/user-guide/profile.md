# MindSpeed 中采集Profile数据

📝 MindSpeed 支持命令式开启Profile采集数据，命令配置介绍如下：

| 配置命令                    | 命令含义                                                                                                    | 
|-------------------------|---------------------------------------------------------------------------------------------------------|
| --profile               | 打开profile开关                                                                                             |
| --profile-step-start    | 配置开始采集步，未配置时默认为10, 配置举例: --profile-step-start 30                                                        |
| --profile-step-end      | 配置结束采集步，未配置时默认为12, 配置举例: --profile-step-end 35                                                          |
| --profile-level         | 配置采集等级，未配置时默认为level0, 可选配置: level0, level1, level2, 配置举例: --profile-level level1                        |
| --profile-with-cpu      | 打开cpu信息采集开关                                                                                             |
| --profile-with-stack    | 打开stack信息采集开关                                                                                           |
| --profile-with-memory   | 打开memory信息采集开关，配置本开关时需打开--profile-with-cpu                                                              |
| --profile-record-shapes | 打开shapes信息采集开关                                                                                          |
| --profile-save-path     | 配置采集信息保存路径, 未配置时默认为./profile_dir, 配置举例: --profile-save-path ./result_dir                                |
| --profile-ranks         | 配置待采集的ranks,未配置时默认为-1，表示采集所有rank的profiling数据，配置举例: --profile-ranks 0 1 2 3, 需注意: 该配置值为每个rank在单机/集群中的全局值 |

---
