### profiling_configs.json规则
1. 可识别字段有```name```, ```tp```, ```cp```, ```pp```, ```seq```, ```experts```, ```ep```, ```mc2```
    - 可识别字段拥有缺省默认值, 可选择不配置
2. ```name```字段默认值为空字符串, 若取值中出现```skip```或其任意大小写变体, 此条配置会被跳过
3. ```tp```字段默认值为"default", 可选配置```mul_t_by=n```, 即开启```tp=默认tp*n```
4. ```seq```字段默认值为"default", 可选配置```slice_seq_by=n```, 即开启```seq_length=默认seq_length//n```, 当```seq_length```低于2K时, 取值变为```默认seq_length*n```
5. 当disable_cp_flag开启时, 开启cp的配置会被跳过
6. 当```seq_length//cp```低于2K时, 此条配置会被跳过
