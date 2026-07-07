# profiling_configs.json rules

1. Recognizable fields include ```name```, ```tp```, ```cp```, ```pp```, ```seq```, ```experts```, ```ep```, ```mc2```
    - Recognizable fields have default values and can be left unconfigured.
2. The ```name``` field default is an empty string. If the value contains ```skip``` or any case variant thereof, this configuration entry will be skipped.
3. The ```tp``` field default is "default", with the optional configuration ```mul_t_by=n```, which enables ```tp=default tp*n```.
4. The ```seq``` field default is "default", with optional configuration ```slice_seq_by=n```, which enables ```seq_length=default seq_length//n```. When ```seq_length``` is below 2K, the value becomes ```default seq_length*n```.
5. When `disable_cp_flag` is enabled, configurations with `cp` enabled will be skipped.
6. When ```seq_length//cp``` is below 2K, this configuration will be skipped.
