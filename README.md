# Diagonal-Tiled Mixed-Precision Attention

Official implementation of **Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference** accepted by EDEG@CVPR 2026. [[Paper](https://arxiv.org/abs/2604.03950)]

This repository provides Triton kernels and evaluation scripts for efficient low-bit mixed-precision attention using the microscaling floating-point format. The proposed **Diagonal-Tiled Mixed-Precision Attention** method accelerates Transformer inference by combining tiling-level mixed-precision computation, kernel fusion, and hardware-aware memory optimization.

## Overview

Transformer-based large language models have strong performance across many real-world tasks, but their inference cost remains high due to the quadratic complexity of attention and the memory bandwidth overhead of high-precision computation.

This project implements a low-bit mixed-precision attention kernel based on the **MXFP** data format. The kernel is designed for next-generation GPU architectures and targets efficient LLM inference with minimal generation-quality degradation.

The core idea is to perform attention computation with mixed low-bit precision at the tile level. By using a diagonal-tiled computation pattern, the kernel improves data reuse, reduces memory traffic, and enables efficient fused attention execution.

## Key Features

* **Low-bit MXFP attention** for efficient Transformer inference.
* **Diagonal-tiled mixed-precision computation** at the attention-tile level.
* **Fused Triton kernel** implementation for reduced memory overhead.
* **Hardware-aware optimization** for modern GPU architectures.
* **Efficient inference** with negligible generation-quality degradation.
* **Evaluation support** for measuring kernel speed and model-level performance.

## Method

Diagonal-Tiled Mixed-Precision Attention combines two forms of low-bit computation inside a fused attention kernel.

The method includes:

1. **MXFP-based low-bit representation**
   Uses microscaling floating-point formats to reduce memory bandwidth and computation cost.

2. **Tile-level mixed precision**
   Applies different low-bit computation modes across attention tiles to balance efficiency and numerical stability.

3. **Diagonal-tiled attention layout**
   Organizes attention computation along diagonal tile regions to improve parallelism and memory efficiency.

4. **Triton kernel fusion**
   Fuses attention operations into a single optimized kernel to reduce intermediate memory movement.

## Repository Structure

```text
.
├── kernels/              # Triton kernels for mixed-precision attention
├── benchmarks/           # Kernel-level benchmarking scripts
├── eval/                 # Model-level evaluation scripts
├── examples/             # Example usage
├── tests/                # Correctness tests
└── README.md
```

The exact directory structure may vary depending on the released implementation.

## Installation

```bash
git clone https://github.com/yifu-ding/MP-Sparse-Attn.git
cd MP-Sparse-Attn

conda create -n dma python=3.10
conda activate dma

pip install -r requirements.txt
```

Recommended dependencies:

```bash
pip install torch triton transformers accelerate
```

## Usage

Example usage:

```python
from dma import diagonal_tiled_attention

output = diagonal_tiled_attention(
    q,
    k,
    v,
    causal=True,
)
```

For benchmarking:

```bash
python benchmarks/benchmark_attention.py
```

For model-level evaluation:

```bash
python eval/evaluate.py
```

Please refer to the scripts in `examples/`, `benchmarks/`, and `eval/` for detailed usage.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{ding2026diagonal,
  title={Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference},
  author={Ding, Yifu and Zhang, Xinhao and Guo, Jinyang},
  booktitle={Proceedings of the CVPR Workshop on Efficient Deep Learning for Edge Computing},
  year={2026}
} 
```

## Acknowledgements

This implementation is built with Triton and PyTorch. We thank the open-source community for providing efficient tools for GPU kernel development and LLM inference research. We also acknowledge prior sparse attention implementations, including [SpargeAttn](https://github.com/thu-ml/SpargeAttn) and [SparseAttention](https://github.com/kyegomez/SparseAttention), which provided useful references for attention kernel development. 
