# Deterministic Computing

## Background and Challenges

During training, various random factors can cause the training process to be inconsistent across experiments, preventing their LOSS curves and performance curves from perfectly overlapping. However, repeatable and comparative experiments sometimes require deterministic computing results to ensure reproducibility.

## Solution

To meet this requirement, the deterministic computing feature has been introduced. It allows users to ensure the consistency of training results across multiple runs through Ascend chips, thereby aiding tasks such as performance tuning and controlled experiments.

## Application Scenario

When you need to conduct performance comparisons or reproduce specific scenarios.

## Usage

To enable this feature, add `--npu-deterministic` to your script.

## Application Effects

The deterministic computing feature ensures that multiple experiments with the same parameters produce identical results.
