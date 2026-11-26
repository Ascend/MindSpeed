# MoE Fixed Router

Assigns experts through a deterministic round-robin rule, for MoE training configurations with fixed routing.

## Usage

Add `--fix-router`. Disabled by default. `--expert-model-parallel-size` must be greater than 1, and `--moe-router-fusion` cannot be enabled at the same time.
