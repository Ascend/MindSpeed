# Feature Precision Guarding System Use Case Execution and Addition Instructions

This document introduces the execution and construction of precision guarding use cases for MindSpeed+Megatron features.

## Feature Precision Guarding Use Case Execution

Take the precision guarding of the Llama model as an example.

### 0. Set up the environment

Refer to the MindSpeed environment installation guide.

### 1. Prepare the tokenizer and dataset

Download the tokenizer for the Llama2 model.

```shell
mkdir ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
cd ../../
```

Download the dataset for Llama2 and process it.

```shell
# Download data
cd ./dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
# Process data
mkdir ./dataset/llama-2-7b-hf/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./dataset/llama-2-7b-hf/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

### 2. Prepare initial parameters

Prepare a set of parameters with TP1PP1 configuration and save them in the `./ckpt_llama` directory (you can randomly initialize and then save the parameters).

### 3. Configure specified parameters

Based on the configuration of the training script `pretrain_fpg_llama.sh`, configure the specified parameters in `fpg_llama_usecase.yaml` as shown below:

```yaml
spec:
  data_path: /home/dataset/llama2/alpaca_text_document
  tokenizer_model: /home/dataset/model/llama-2-7b-hf/tokenizer.model
  mbs: 2 # MBS   micro-batch-size
  gbs: 16 # GBS  global-batch-size
  train_iters: 5000
```

### 4. Configure baseline parameters and feature parameters

Refer to the fgp_llama_usecase.yaml configuration. Each system use case includes pre_process and run steps. Each step can be configured with multiple processing instances, and each processing instance includes two configurations: script_file and param.
**For example, the two steps of baseline**

- pre_process : Prepare the initialization parameters for the baseline
- run : Execute the script to obtain the loss baseline

The configuration rules for each feature are consistent with those of the baseline. For execution logic, see pretrain_gpt_fpg.py.
**Execution notes**

- Entry file: `pretrain_gpt_fpg.py`, executed by `python MindSpeed/tests_extend/system_tests/feature_precision_guarding/pretrain_gpt_fpg.py`
- Logs are stored in `./{%YU_%m_%d}logs`, with the log file naming convention: `{feature_name}-{stage_name}-{state_index}-{timestamp}.log`
- After the run instance completes, the loss is automatically compared with the baseline, and the results are written to the `report.csv` file

## Adding Feature Precision Guarding Use Cases

### Method 1: Add use cases in the configuration file

Example: Verify the precision of MC2 + Distributed Optimizer
**Analysis**

- MC2 requires setting the parameter `--use-ascend-mc2`. MC2 depends on TP and sequence parallelism (SP base script already configured)
- Distributed optimizer requires adding the extra parameter `--use-distributed-optimizer`
Add one entry under features in `fpg_llama_usecase.yaml`, as shown below:

```yaml
  - mc2_distributed-optim:
      pre_process:
        - { script_file: llama_param_cvt.sh,
            param: {
              py_script_path: MindSpeed/tests_extend/system_tests/feature_precision_guarding/llama_param_cvt.py,
              input_model_dir: ./ckpt_llama,
              output_model_dir: ./ckpt_llama_tp2pp2,
              tp_size: 2,
              pp_size: 2,
              num_layers: 12,
              num_heads: 8,
              hidden_size: 512
            }
        }
      run:
        - { script_file: pretrain_fpg_llama.sh,
            param: { npu_per_node: 8,
                     tp_size: 2,
                     pp_size: 2,
                     load_ckpt_dir: ./ckpt_llama_tp2pp2,
                     extra_args: '"--use-distributed-optimizer"',
                     ascend_mc2: 1
            }
        }
```

Where `num_per_node`, the number of cards used per node, is required.
`tp_size` and `pp_size` are required. It has been verified that the loss at 5k steps for TP1PP1 and the mean absolute/relative error for TP2PP2 are less than 0.5%. TP1PP1 can be used as the baseline, or you can modify the TP and PP settings in the baseline;

### Method 2: Create a new use case file

Create a new use case file following the format of `fpg_llama_usecase.yaml`, for example, `fpg_qwen_usecase.yaml`.
Then add a line in the `main` method of `pretrain_gpt_fpg.py`, for example:

```python
    xtest_pretrain_fpg("fpg_qwen_usecase.yaml")
```
