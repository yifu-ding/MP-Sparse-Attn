## Command-Line Argument Reference

This program's command-line arguments are grouped by functionality as follows:

### 1. Model Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | `str` | `None` | Model path or model identifier. |
| `--device` | `str` | `'cuda'` | Runtime device. Options: `'cuda'` or `'cpu'`. |

### 2. Sparse Attention Configuration (for the codebase baseline SpargeAttn; not used by our method)

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--tune` | `store_true` | `False` | Enable operator tuning. |
| `--verbose` | `store_true` | `False` | Print verbose logs. |
| `--l1` | `float` | `0.06` | L1-norm threshold for query-key sparse selection. |
| `--pv_l1` | `float` | `0.065` | L1-norm threshold for product-value sparse selection. |

### 3. Dataset Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--e` | `store_true` | `False` | Evaluate LongBench-E (extended set). |
| `--test_dataset_name` | `str` | `'all'` | Evaluate a single dataset (useful for debugging). Default `'all'` evaluates all LongBench datasets. |

### 4. Output and Logging Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--output_path` | `str` | `'results'` | Output directory for results. |
| `--model_out_path` | `str` | `''` | Path to save tuned model `state_dict` (SpargeAttn only). |
| `--use_wandb` | `store_true` | `False` | Enable Weights & Biases logging. |
| `--num_fewshots` | `int` | `None` | Evaluate only a specified number of few-shot samples (for debugging or quick checks). |

### 5. Evaluation Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--test_speedup` | `store_true` | `False` | Run inference speed benchmark only (no accuracy evaluation). |
| `--get_pred` | `store_true` | `False` | Generate predictions. |
| `--compute_accuracy` | `store_true` | `False` | Compute prediction accuracy. |

> Speed-only evaluation: use `--test_speedup`.
> Accuracy evaluation:
> Method 1: enable both `--get_pred` and `--compute_accuracy`.
> Method 2: run `--get_pred` first, then run `--compute_accuracy` separately.

### 6. Method Selection

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--kernel_name` | `str` | `None` | Attention kernel type to use. Options:<br> - `online_routing`: legacy method (deprecated)<br> - `mxfp_attn`: production Triton kernel<br> - `mxfp_attn_debug`: torch+triton debug implementation for algorithm iteration and quick validation; absolute accuracy is not reliable, use relative comparisons only<br> - `native`: PyTorch native SDPA attention<br> - `spargeattn`: built-in baseline in this codebase |
| `--mxfp_bw` | `str` | `'mxfp8'` | Bit-width setting for low-bit attention; affects quantization and compute precision. Options:<br> - `mxfp4`, `mxfp8`, `nvfp4`: single-bitwidth schemes<br> - `mxfp8_diag`: hybrid scheme using `mxfp8` on diagonal and `nvfp4` off-diagonal (under validation) |

### 7. Other Tricks

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--skip_thresh` | `float` | `None` | Deprecated legacy skip-strategy threshold; currently unused. |
| `--smooth_k` | `store_true` | `False` | Enable key-matrix smoothing trick. |
| `--dual_scale` | `store_true` | `False` | Enable dual-scale technique (an optimization used in our method). |
