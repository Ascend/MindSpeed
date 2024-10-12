# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import glob
import copy
import warnings
import ast
from typing import Optional

import pandas as pd
import numpy as np
import gpytorch
import torch

from mindspeed.core.auto_parallel import (
    ARD_NUM_DIMS,
    KeyField,
    sample_cache,
    model_manager
)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, operator, train_inputs=None,
                 train_targets=None, raw_lengthscale=None,
                 likelihood=None, dtype=torch.float64):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood=likelihood)
        self.operator = operator
        self.dtype = dtype

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ARD_NUM_DIMS[operator],
                                          lengthscale_constraint=gpytorch.constraints.GreaterThan(3e-2)))
        if raw_lengthscale is not None:
            self.covar_module.base_kernel.raw_lengthscale.data \
                = self.raw_lengthscale * torch.ones_like(self.covar_module.base_kernel.raw_lengthscale.data)

        self.train_round = 0
        self.train_data = pd.DataFrame()

        self.y_train_mean: Optional[torch.Tensor] = None
        self.y_train_std: Optional[torch.Tensor] = None
        self.x_train_std: Optional[torch.Tensor] = None

    def get_model_info(self):
        return self.train_data, self.train_round

    def set_model_info(self, values):
        self.train_data, self.train_round = values
        # set model info by train_data
        self.data_standardize()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def fit(self, profiling_file, multi_operator_data, num_iter=3000, lr=0.03):
        hd = DataHandler(profiling_file, multi_operator_data)
        data = hd.generate_data(self.operator)
        # merge self.train_data with new train_data
        self.update_data(data)
        # set model train_inputs and target_inputs
        self.data_standardize()
        # clear cache
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        for i in range(num_iter):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            if i % 100 == 0:
                logs = 'Iter %d/%d - Loss: %.5f   outputscale: %.5f   noise: %.5f' % (
                    i + 1, num_iter, loss.item(),
                    self.covar_module.outputscale.item(),
                    self.likelihood.noise.item()
                ) + '   lengthscale: ' + str(
                    np.round(self.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0], 5))
                print(logs)
            optimizer.step()
        self.eval()
        self.likelihood.eval()
        self.train_round += 1

    def update_data(self, data: pd.DataFrame):
        """
        :param data columns = [shape error count]
        """
        if not self.train_data.empty:
            exits_shapes = self.train_data.loc[:, KeyField.InputShapes].values.tolist()
            for index, rows in data.iterrows():
                shape = getattr(rows, KeyField.InputShapes)
                # update existent input_shape
                if shape in exits_shapes:
                    error, number = data[data[KeyField.InputShapes] == shape].iloc[:, 1:3].values.flatten()
                    current_train_data = self.train_data[self.train_data[KeyField.InputShapes] == shape]
                    train_error, train_number = current_train_data.iloc[:, 1:3].values.flatten()
                    count = int(number + train_number)
                    new_error = (error * number + train_error * train_number) / count
                    self.train_data[self.train_data[KeyField.InputShapes] == shape] = [shape, new_error, count]
                else:
                    # save new input_shape
                    self.train_data = pd.concat([self.train_data, rows.to_frame().T], ignore_index=True)
        else:
            self.train_data = data

    def data_standardize(self):
        y_train = torch.tensor(self.train_data['error'], dtype=self.dtype)
        x_train = self.train_data[KeyField.InputShapes].str.split(',', expand=True).values.astype(int)
        x_train = torch.tensor(x_train, dtype=self.dtype).log()
        if x_train.shape[0] == 1:
            self.x_train_std = torch.tensor(np.ones(x_train.shape), dtype=self.dtype)
            self.y_train_std = torch.tensor(1, dtype=self.dtype)
        else:
            self.x_train_std, self.y_train_std = torch.std(x_train, dim=0), torch.std(y_train, dim=0)
            self.x_train_std[self.x_train_std == 0] = 1.
            self.y_train_std[self.y_train_std == 0] = 1.
        x_train /= self.x_train_std
        self.y_train_mean = torch.mean(y_train, dim=0)
        y_train = (y_train - self.y_train_mean) / self.y_train_std
        self.set_train_data(x_train, y_train, strict=False)


class Sampler:
    def __init__(self, num_sample=10, pre_thd=0):
        self.pre_thd = pre_thd
        self.num_sample = torch.Size([num_sample])

    def run(self, operator, direct_time, output_shape: list, *input_shape):
        input_shape = copy.deepcopy(input_shape)
        output_shape = copy.deepcopy(output_shape)
        # modify input_shape
        input_shape = Sampler.reduce_dim(operator, output_shape, input_shape)
        # check cache
        cached_samples = getattr(sample_cache, operator)
        sample = cached_samples.get(input_shape, None)
        if sample is not None:
            return sample
        # load model
        model = model_manager.get_cached_model(operator)
        # predict
        input_shape_np = np.array(input_shape).reshape(1, -1)
        fixed_shape = np.concatenate([input_shape_np, input_shape_np], axis=0)
        x = torch.tensor(fixed_shape, dtype=torch.float64).log()
        if model is None:
            relative_error = np.zeros(self.num_sample)
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = model(x / model.x_train_std)
            pred = pred * model.y_train_std.item() + model.y_train_mean.item()
        relative_error = pred.sample(self.num_sample).cpu().numpy()[:, 0]
        sample = direct_time * (relative_error + 1.).flatten()
        negative_indices = np.where(sample <= self.pre_thd)[0]
        if negative_indices.size > 0:
            sample[negative_indices] = 0
            warnings.warn(f'Uncertainty of {operator} is too large, input shape: {input_shape}', Warning)
        # save prediction data
        cached_samples[input_shape] = sample
        return sample

    @staticmethod
    def reduce_dim(operator, output_shape, input_shapes):
        input_shapes = copy.deepcopy(input_shapes)
        output_shape = copy.deepcopy(output_shape)
        if operator in ['LayerNorm', 'LayerNormGrad']:
            input_shape = input_shapes[0]
        elif operator in ['FastGelu', 'FastGeluGrad']:
            input_shape = output_shape
        elif operator in ['Softmax', 'SoftmaxGrad']:
            input_shape = output_shape
        elif operator == 'Add' or operator == 'Mul':
            if len(input_shapes[0]) >= len(input_shapes[1]):
                max_dims, min_dims = input_shapes
            else:
                min_dims, max_dims = input_shapes
            if len(max_dims) == 2:
                max_dims.insert(0, 1)
            if len(max_dims) == 1:
                max_dims = [1, 1, max_dims[0]]
            if len(min_dims) == 3:
                min_dims = [1, 1, 1]
            elif len(min_dims) == 2:
                min_dims = [2, 1, 1]
            else:
                min_dims = [2, 2, 1]
            max_dims.extend(min_dims)
            input_shape = max_dims
        elif operator == 'BatchMatMul':
            if len(input_shapes) != 2:
                raise AssertionError(f"Dim of BatchMatMul is {len(input_shapes)}")
            b, k, m = output_shape[0], output_shape[2], output_shape[1]
            n = input_shapes[0][1:] + input_shapes[1][1:]
            for shape in output_shape[1:]:
                n.remove(shape)
            input_shape = [b, m, n[0], k]
        elif operator == 'MatMul':
            if len(input_shapes) != 2:
                raise AssertionError(f"Dim of MatMul is {len(input_shapes)}")
            input_shape = input_shapes[0]
            input_shape.extend(input_shapes[1])
            for shape in output_shape:
                input_shape.remove(shape)
            output_shape.insert(1, input_shape[0])
            input_shape = output_shape
        elif operator == 'RmsNorm' or operator == 'RmsNormGrad':
            input_shape = input_shapes[0]
        elif operator == 'FlashAttentionScore' or operator == 'FlashAttentionScoreGrad':
            input_shape = input_shapes[0]
        else:
            raise ValueError(f"{operator} not supported.")

        return tuple(input_shape)


class DataHandler:
    def __init__(self, profiling_file, multi_operator_data: pd.DataFrame):
        self.sample_data = multi_operator_data
        self.profiling = self.extract_target_data(profiling_file)
        self.current_profiling_operator = None
        self.current_sample_operator = None
        self.backward_flag = False

    @staticmethod
    def extract_target_data(file):
        if os.path.isdir(file):
            file = glob.glob(os.path.join(file, "*.csv"))
            data = pd.concat((pd.read_csv(f) for f in file), ignore_index=True).loc[:,
                   [KeyField.OpType, KeyField.InputShapes, KeyField.OutputShapes, KeyField.Duration]]
        else:
            data = pd.read_csv(file).loc[:,
                   [KeyField.OpType, KeyField.InputShapes, KeyField.OutputShapes, KeyField.Duration]]
        data.loc[data['Type'].str.startswith('MatMul'), 'Type'] = 'MatMul'
        data.loc[data['Type'].str.startswith('BatchMatMul'), 'Type'] = 'BatchMatMul'
        data.loc[
            (data['Type'].str.startswith('LayerNorm') & 
            ~(data['Type'].str.contains('Back') | data['Type'].str.contains('Grad'))), 'Type'
        ] = 'LayerNorm'
        data.loc[
            (data['Type'].str.startswith('LayerNorm') & 
            (data['Type'].str.contains('Back') | data['Type'].str.contains('Grad'))), 'Type'
        ] = 'LayerNormGrad'
        # filter
        data = data[(data[KeyField.Duration] > 5) & (data[KeyField.InputShapes].str.len() > 4)].reset_index(drop=True)
        return data

    @staticmethod
    def convert_dim(data):
        new_input_shape = []
        for index, tmp_data in data[[KeyField.OpType, KeyField.InputShapes, KeyField.OutputShapes]].iterrows():
            op, input_shape, output_shape = tmp_data.tolist()
            input_shape, output_shape = ast.literal_eval(input_shape), ast.literal_eval(output_shape)
            if op == 'LayerNorm' or op == 'LayerNormGrad':
                input_shape = input_shape.split(';')[0]
            elif op == 'Add' or op == 'Mul':
                dims = input_shape.split(';')
                d0_l, d1_l = dims[0].split(','), dims[1].split(',')
                if len(d0_l) >= len(d1_l):
                    max_length_dim = d0_l
                    min_length_dim = d1_l
                else:
                    max_length_dim = d1_l
                    min_length_dim = d0_l
                if len(max_length_dim) == 2:
                    max_length_dim = ['1', '1', max_length_dim[0], max_length_dim[1]]
                elif len(max_length_dim) == 1:
                    max_length_dim = ['1', '1', '1', max_length_dim[0]]
                elif len(max_length_dim) == 3:
                    max_length_dim.insert(0, '1')
                if len(min_length_dim) == 3:
                    min_length_dim = ['2', '1', '1', '1']
                elif len(min_length_dim) == 2:
                    min_length_dim = ['2', '2', '1', '1']
                elif len(min_length_dim) == 1:
                    min_length_dim = ['2', '2', '2', '1']
                elif len(min_length_dim) == 4:
                    min_length_dim = ['1', '1', '1', '1']
                max_length_dim.extend(min_length_dim)
                input_shape = ','.join(max_length_dim)
            elif op == 'BatchMatMul':
                output_shape = output_shape.split(',')
                b, k, m = output_shape[0], output_shape[2], output_shape[1]
                input_shapes = input_shape.split(';')
                n = input_shapes[0].split(',')[1:] + input_shapes[1].split(',')[1:]
                for shape in output_shape[1:]:
                    n.remove(shape)
                input_shape = ','.join([b, m, n[0], k])
            elif op == 'MatMul':
                input_shape = input_shape.replace(';', ',').split(',')
                output_shape = output_shape.split(',')
                for shape in output_shape:
                    input_shape.remove(shape)
                output_shape.insert(1, input_shape[0])
                input_shape = ','.join(output_shape)
            elif op == 'Softmax' or op.startswith('SoftmaxGrad'):
                input_shape = input_shape.split(';')[0]
            elif op == 'RmsNorm' or op == 'RmsNormGrad':
                input_shape = input_shape.split(';')[0]
            elif op == 'FlashAttentionScore' or op == 'FlashAttentionScoreGrad':
                input_shape = input_shape.split(';')[0]
            else:
                raise TypeError(f"{op} don't support")
            new_input_shape.append(input_shape)
        return new_input_shape

    def handle_transpose(self):
        input_shapes = []
        for index, sample in self.current_profiling_operator.iterrows():
            input_shape = sample[KeyField.InputShapes]
            input_shape = ast.literal_eval(input_shape).split(';')
            input_shape = [list(map(lambda x: int(x), s.split(','))) for s in input_shape]
            output_shape = ast.literal_eval(sample[KeyField.OutputShapes]).split(',')
            output_shape = [int(s) for s in output_shape]
            if sample[KeyField.OpType] == 'BatchMatMul':
                if output_shape[1] != input_shape[0][1]:
                    input_shape[0][1], input_shape[0][2] = input_shape[0][2], input_shape[0][1]
                if output_shape[-1] != input_shape[1][-1]:
                    input_shape[1][1], input_shape[1][2] = input_shape[1][2], input_shape[1][1]
            elif sample[KeyField.OpType] == 'MatMul':
                if output_shape[0] != input_shape[0][0]:
                    input_shape[0][0], input_shape[0][1] = input_shape[0][1], input_shape[0][0]
                if output_shape[-1] != input_shape[1][-1]:
                    input_shape[1][0], input_shape[1][1] = input_shape[1][1], input_shape[1][0]
            input_shape1 = ','.join([str(i) for i in input_shape[0]])
            input_shape2 = ','.join([str(i) for i in input_shape[1]])
            input_shape_sum = input_shape1 + ';' + input_shape2
            input_shapes.append(f'"{input_shape_sum}"')
        self.current_profiling_operator.loc[:, KeyField.InputShapes] = input_shapes

    def handle_layer_norm_backward(self, operator):
        profiling = self.profiling[self.profiling[KeyField.OpType] == operator].reset_index(drop=True)
        back_grad_data = pd.DataFrame()
        for index in range(0, profiling.shape[0], 2):
            sum_duration = profiling.loc[index, KeyField.Duration] + profiling.loc[
                index + 1, KeyField.Duration]
            input_shape = profiling.loc[index, KeyField.InputShapes].split(';')[0] + '"'
            back_grad_data.loc[index, KeyField.OpType] = 'LayerNormGrad'
            back_grad_data.loc[index, KeyField.InputShapes] = input_shape
            back_grad_data.loc[index, KeyField.OutputShapes] = input_shape
            back_grad_data.loc[index, KeyField.Duration] = sum_duration
        return back_grad_data.reset_index(drop=True)

    def handle_fv(self):
        condition = self.current_profiling_operator[KeyField.InputShapes].str.replace('"', '').str.split(';').map(
            lambda x: x[:3]).map(lambda x: x[0] == x[1] == x[2])
        self.current_profiling_operator = self.current_profiling_operator[condition]
        # 对FV_grad的input_shape可能出现的异常情况容错处理
        target_shape = self.current_sample_operator[KeyField.InputShapes].values[0]
        current_shape = self.current_profiling_operator[KeyField.InputShapes].values[0]
        if target_shape.split(';')[1] != current_shape.split(';')[1]:
            self.current_profiling_operator[KeyField.InputShapes] = target_shape
            
    def generate_data(self, operator):
        # 串行处理各个算子
        if len(operator) == 2:
            # layer_norm反向特殊处理
            self.current_profiling_operator = self.handle_layer_norm_backward(operator)
            operator = self.current_profiling_operator.loc[0][KeyField.OpType]
        else:
            self.current_profiling_operator = self.profiling[self.profiling[KeyField.OpType] == operator]
            self.backward_flag = False
        if operator.endswith('Grad'):
            self.backward_flag = True
            operator = operator.split('Grad')[0]
            # matmul和batch_matmul需要考虑转置情况
        if operator in ['MatMul', 'BatchMatMul']:
            self.handle_transpose()
        # convert sample input_shape
        self.current_sample_operator = self.sample_data[
            self.sample_data[KeyField.OpType].str.startswith(operator)].reset_index(
            drop=True)
        # 删除负载均衡产生的shape和对FVGrad可能出现的异常Input_shape容错处理.
        if operator.startswith('FlashAttention'):
            self.handle_fv()
        # convert profiling input_shape
        self.current_profiling_operator.loc[:, KeyField.InputShapes] = self.convert_dim(
            self.current_profiling_operator
        )
        self.current_sample_operator[KeyField.InputShapes] = self.convert_dim(self.current_sample_operator)
        # 获取当前算子的所有input_shape
        set_operator = self.current_sample_operator[KeyField.InputShapes].drop_duplicates().tolist()
        errors_df = pd.DataFrame()
        # 计算每个input_shape的相对误差
        for shape in set_operator:
            # 获取profiling数据当前input_shape的所有样本
            tmp_data = self.current_profiling_operator[
                self.current_profiling_operator[KeyField.InputShapes] == shape].copy()
            if self.backward_flag:
                direct_mean = self.current_sample_operator[
                    self.current_sample_operator[KeyField.InputShapes] == shape
                ]['bwd_time'].values[0]
            else:
                direct_mean = self.current_sample_operator[
                    self.current_sample_operator[KeyField.InputShapes] == shape
                ]['fwd_time'].values[0]
            # 计算相对误差
            tmp_data['error'] = (tmp_data[KeyField.Duration] - direct_mean) / direct_mean
            tmp_data['direct_mean'] = direct_mean
            errors_df = pd.concat([errors_df, tmp_data], axis=0)
        if errors_df.empty:
            raise AssertionError('profiling_shape mismatch operator_shape')

        # 分组平均和计数
        train_data = errors_df.groupby(KeyField.InputShapes).agg(
            {'error': 'mean', KeyField.InputShapes: 'count'})
        train_data.rename(columns={KeyField.InputShapes: 'sample_number'}, inplace=True)
        train_data.reset_index(inplace=True)
        return train_data