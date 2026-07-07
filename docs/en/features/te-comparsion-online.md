# Low-Precision Online Comparison

## Problem Analysis

Precision issues are often difficult to locate in model training. To improve the efficiency of precision tracking and ensure the reliability of low-precision modules, a method for real-time monitoring of low-precision computation results is needed, while also reducing the cost of issue localization.

## Solution

The low-precision online comparison feature helps users inspect the precision differences of low-precision operations (including cast and quantmatmul) in real time by comparing computation results across different hardware (NPU, CPU) and different precisions (BF16). When this feature is enabled, computations are performed on both NPU and CPU/BF16 simultaneously, and the error values are output, allowing users to detect potential precision issues conveniently.

## Application Scenario

This feature is applicable to models that have low-precision training enabled.

## Usage

1. Add `--te-comparison-with-cpu` to the parameter list in the training script to enable low-precision online comparison using CPU computation results as the accuracy benchmark.
2. Add `--te-comparison-with-bf16` to the parameter list in the training script to enable low-precision online comparison using BF16 computation results as the accuracy benchmark.

## Application Effects

During low-precision computation, the error between the cast and quantmatmul operations on the NPU and the accuracy benchmark (CPU or BF16) will be output at each step. If the error exceeds the preset range, an error will be reported and the training process will be terminated.
