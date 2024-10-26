import ast
import json
import os
import re
import time
import stat
import numpy as np
import yaml


def load_config(conf_yaml):
    """
    Read the case YAML file.
    @param file_path: file path
    @return:
    """
    with open(file=conf_yaml, mode='r', encoding='utf-8') as f:
        crf = f.read()
        yaml_data = yaml.safe_load(stream=crf)
        return yaml_data


def write_file(file_path, write_content: str):
    flags = os.O_RDWR | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP
    with os.fdopen(os.open(file_path, flags, modes),
                   'a') as fout:
        fout.write(write_content)
        fout.write("\n")


def run_script_instance(script_conf: dict, spec_param: dict, logs_dir: str, log_name: str):
    script_file = script_conf['script_file']
    param = dict(script_conf['param'])

    # Preparing Case Parameters
    param_list = [f"{str(key).upper()}={str(value)}" for key, value in param.items()]
    param_str = " ".join(param_list)
    script_path = os.path.join(os.path.dirname(__file__), script_file)
    print("=================== STARTING ======================")
    print(f"param : {param_str}")

    # The configuration of the use case takes precedence over the specified configuration.
    # (If the parameter configured in the case is deleted from the specified parameter)
    [spec_param.pop(k) for k in param.keys() if k in spec_param.keys()]
    sepc_param = " ".join([f"{str(key).upper()}={str(value)}" for key, value in spec_param.items()])
    print(f"spec_param : {sepc_param}")

    # Create a log directory.
    os.makedirs(logs_dir, exist_ok=True)
    log_save_path = os.path.join(logs_dir, f"{log_name}.log")

    # run usecase
    cmd = f"sh {script_path} {param_str} {sepc_param} 2>&1 | tee {log_save_path}"
    os.system(cmd)
    print("==================== ENDING =======================")


def compare_loss_with_baseline(feat_log_name_prefix: str, baseline_log_name_prefix: str, logs_dir: str) -> dict:
    def _get_target_log_path(_name_prefix: str, _logs_dir: str):
        log_files = sorted([log_file for log_file in os.listdir(_logs_dir) if log_file.startswith(_name_prefix)])

        if len(log_files) < 1:
            return {"err_msg": f"{_name_prefix} : no log file found."}

        log_file = log_files[-1]  # take the latest one
        log_path = os.path.join(logs_dir, log_file)
        return log_path

    def _get_log_data_list(_log_file):
        """
        parse megatron log file
        """
        loss_dict = {}
        with open(_log_file, 'rb') as f:
            file_lines = f.readlines()

        file_lines = [str(line.decode(encoding='utf-8')).strip() for line in file_lines]
        for line in file_lines:
            if 'iteration' in line and 'finished' not in line:
                try:
                    iteration = int(re.findall('iteration\s+(.*?)\/ ', line)[0])
                    loss = ast.literal_eval(re.findall('lm loss: (.*?) ', line)[0])
                    loss_dict[iteration] = loss
                except:
                    print(f"failed to parse line : {line}")
                    continue
        return {"loss_dict": loss_dict}

    def _get_compare_infos(datas1: dict, datas2: dict, max_step, commp_func):
        metrics = []
        step_num = min(max(datas1.keys()), max(datas2.keys()))
        for i in range(0, min(max_step, step_num)):
            v1 = datas1.get(i, None)
            if v1 is None:
                continue

            v2 = datas2.get(i, None)
            if v2 is None:
                continue

            metric_val = commp_func(v1, v2) if v1 > 0 and v2 > 0 else 0
            metrics.append(metric_val)
        return metrics

    # step 1:  get log path
    feat_logpath = _get_target_log_path(_name_prefix=feat_log_name_prefix, _logs_dir=logs_dir)
    base_logpath = _get_target_log_path(_name_prefix=baseline_log_name_prefix, _logs_dir=logs_dir)
    if isinstance(feat_logpath, dict):  # missing base log file
        return feat_logpath
    if isinstance(base_logpath, dict):  # missing base log file
        return base_logpath

    # step 2: get loss data
    feat_loss_dict = _get_log_data_list(_log_file=feat_logpath)['loss_dict']
    base_loss_dict = _get_log_data_list(_log_file=base_logpath)['loss_dict']

    if len(feat_loss_dict) < 1 or len(base_loss_dict) < 1:
        return {"err_msg": f"The number of loss steps is empty. "
                           f"{feat_log_name_prefix}:{len(feat_loss_dict)}, {baseline_log_name_prefix}:{len(base_loss_dict)}"}

    # step 3: compare loss
    abs_metrics = _get_compare_infos(datas1=base_loss_dict, datas2=feat_loss_dict, max_step=10000,
                                     commp_func=lambda v1, v2: abs(v1 - v2))
    rel_metrics = _get_compare_infos(datas1=base_loss_dict, datas2=feat_loss_dict, max_step=10000,
                                     commp_func=lambda v1, v2: abs(v1 - v2) / (abs(v1) + 1e-9))
    if len(abs_metrics) < 1 or len(rel_metrics) < 1:
        return {"err_msg": "comparable data is empty."}

    return {"compare_step_num": len(abs_metrics), "MRE": np.mean(rel_metrics), "MaxRE": np.max(rel_metrics),
            "MAE": np.mean(abs_metrics), "MaxAE": np.max(abs_metrics)}


def run_feature_instance(feat_name: str, feat_conf: dict, spec_param: dict, logs_dir: str):
    process_flow = ['pre_process', 'run']
    for stage in process_flow:
        print(f"==================== {stage} -start =======================")
        if stage in feat_conf.keys() and feat_conf[stage] is not None:
            for i, script_conf in enumerate(feat_conf[stage]):
                log_name_prefix = f"{feat_name}-{stage}-{str(i).zfill(2)}"
                if 'script_file' not in script_conf:
                    continue
                run_script_instance(script_conf=script_conf,
                                    spec_param=spec_param,
                                    logs_dir=logs_dir,
                                    log_name=log_name_prefix)

                if stage == 'run' and feat_name != "baseline":
                    baseline_log_name_prefix = "baseline-run-00"
                    msg = compare_loss_with_baseline(feat_log_name_prefix=log_name_prefix,
                                                     baseline_log_name_prefix=baseline_log_name_prefix,
                                                     logs_dir=logs_dir)
                    report_info = {"time": f"{time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(int(time.time())))}",
                                   f"{log_name_prefix} vs {baseline_log_name_prefix}": msg}
                    report_file = os.path.join(logs_dir, "report.csv")
                    write_file(report_file, json.dumps(report_info))

        print(f"==================== {stage} -end =======================")


def xtest_pretrain_fpg(usecase_yaml):
    # step1 : Loading configuration parameters
    conf_yaml = os.path.join(os.path.dirname(__file__), usecase_yaml)
    conf_data = load_config(conf_yaml=conf_yaml)

    # Specified parameters
    spec_param_dict = dict(conf_data['spec'])

    # step2 : Run the baseline script to obtain the baseline training logs.
    logs_dir = f"./{time.strftime('%Y_%m_%d', time.localtime(int(time.time())))}logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    print(conf_data['run_baseline'])
    if bool(conf_data['run_baseline']):
        run_feature_instance(feat_name="baseline",
                             feat_conf=dict(conf_data['baseline']),
                             spec_param=spec_param_dict,
                             logs_dir=logs_dir)

    # step3 : Run All Features
    feat_conf_list = conf_data['features']
    for feat_info in feat_conf_list:
        for feat_name, feat_conf in dict(feat_info).items():
            run_feature_instance(feat_name=feat_name,
                                 feat_conf=dict(feat_conf),
                                 spec_param=spec_param_dict,
                                 logs_dir=logs_dir)


if __name__ == "__main__":
    xtest_pretrain_fpg("fpg_llama_usecase.yaml")
