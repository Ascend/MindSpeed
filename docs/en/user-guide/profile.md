# Data Profiling in MindSpeed

📝 MindSpeed supports enabling of data profiling through commands. The command configurations are as follows:

| Command Configuration    | Command Description                                                                                                                                                                                                 |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --profile               | Enables the profiling switch                                                                                                                                                                                          |
| --profile-step-start    | Configures the start step for profiling. When not configured, it defaults to 10. Configuration example: `--profile-step-start 30`                                                                                   |
| --profile-step-end      | Configures the end step for profiling. When not configured, it defaults to 12. Configuration example: `--profile-step-end 35`                                                                                       |
| --profile-level         | Configures the profiling level. When not configured, it defaults to level0. Optional configurations: level0, level1, level2. Configuration example: `--profile-level level1`                                        |
| --profile-with-cpu      | Enables the CPU information profiling switch                                                                                                                                                                       |
| --profile-with-stack    | Enables the stack information profiling switch                                                                                                                                                                     |
| --profile-with-memory   | Enables the memory information profiling switch. When configuring this switch, `--profile-with-cpu` must be enabled                                                                                                  |
| --profile-record-shapes | Enables the shapes information profiling switch                                                                                                                                                                    |
| --profile-save-path     | Configures the save path for collected information. When not configured, it defaults to ./profile_dir. Configuration example: `--profile-save-path ./result_dir`                                                      |
| --profile-ranks         | Configures the ranks to be collected. When not configured, it defaults to `-1`, indicating that profiling data for all ranks will be collected. Configuration example: `--profile-ranks 0 1 2 3`. Note: This configuration value is the global value for each rank in a single machine/cluster |
