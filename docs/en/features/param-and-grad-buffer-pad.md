# Parameter and Gradient Buffer Alignment

When the distributed optimizer computes the parameter layout, the parameter storage start position is byte-aligned.

## Usage

Add the following to the training configuration:

```shell
--use-distributed-optimizer
--param-and-grad-buffer-pad 512
```

`--param-and-grad-buffer-pad` is not set by default, and its value must be greater than 0; Ascend recommends setting it to 512. It currently cannot be used together with `--use-layer-wise-distributed-optimizer`.
