# FlashMLA

## `precise-flashmla` 精度对齐分支

本分支直接基于 DeepSeek FlashMLA 最新的 `upstream/main`（创建分支时为
`15f13e5030374295491c5ce31b02d7e63a7772c6`），修复 H200 / SM90 sparse MLA
prefill 前向与 TileLang 实现之间的数值偏差。它不包含本仓库私有 `main` 分支上的
vLLM sparse-decode active-head 优化。

### 改动内容

| 项目 | 上游实现 | 本分支实现 |
| --- | --- | --- |
| QK 计算 | 两个 consumer warp group 各自使用 `64x64` WGMMA，独立推进不同的 token tile | 每个 warp group 使用 `64x32` WGMMA，两个 warp group 合作完成同一个 64-token tile |
| Online softmax | 两个 warp group 按不同的 tile 顺序分别更新局部状态 | 每个 64-token tile 内跨 warp group 合并 max 和 sum，再统一更新 softmax 状态 |
| PV 与输出 | 每个 warp group 独立消费自己的概率分块 | 两个 warp group 共享同一逻辑 tile 的概率，各自保留 256 个输出维度，合计写回 `d_v=512` |
| 数学编译选项 | 全局使用 NVCC `--use_fast_math` | 移除 `--use_fast_math`，恢复精确的除法、平方根和次正规数处理；LSE 写回显式关闭 FMA 收缩 |

核心原则是让 FlashMLA 按照与 TileLang 相同的 **64-token 逻辑归约顺序** 更新
online softmax，避免微小的前向误差在 cuDNN DSA backward 和端到端训练中被放大。
CUDA kernel 接口没有变化，修改范围为：

- `csrc/sm90/prefill/sparse/config.h`
- `csrc/sm90/prefill/sparse/phase1.cuh`
- `setup.py`

在 GLM-5.2 训练 shape（Q `[16384, 64, 576]`、KV `[16384, 1, 576]`、
top-k 2048）上，修复后的 FlashMLA 与 TileLang 对比结果为：输出逐比特一致，LSE
最大绝对误差为 `0`，output 和 LSE 均逐 bit 一致。本次修复只改变 sparse MLA forward；训练中 backward
仍由调用方选择，例如 XTuner 当前使用 cuDNN DSA backward。

### H200 / CUDA 12.8 编译

```bash
git switch precise-flashmla
git submodule update --init --recursive

FLASH_MLA_DISABLE_SM100=1 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
python3 setup.py build_ext --inplace
```

也可以安装到当前 Python 环境：

```bash
FLASH_MLA_DISABLE_SM100=1 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
python3 -m pip install -v --no-build-isolation .
```

CUDA 12.8 必须设置 `FLASH_MLA_DISABLE_SM100=1`；SM100 需要 CUDA 12.9 或更新版本，
并且不在本次精度与性能验证范围内。

### 使用与验证

Python 调用方式与上游完全一致：

```python
from flash_mla import flash_mla_sparse_fwd

out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, sm_scale)
```

源码目录内编译后，可运行上游 sparse-prefill 测试：

```bash
python3 tests/test_flash_mla_sparse_prefill.py
```

调用方若此前已经安装过其他 FlashMLA，请确认实际加载的是本分支生成的扩展，避免
源码与 `.so` 混用：

```bash
python3 -c 'import flash_mla.cuda as m; print(m.__file__)'
```

## Introduction

FlashMLA is DeepSeek's library of optimized attention kernels, powering the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) and [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) models. This repository contains the following implementations:

**Sparse Attention Kernels**

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).*

- Token-level sparse attention for the prefill stage
- Token-level sparse attention for the decoding stage, with FP8 KV cache

**Dense Attention Kernels**

- Dense attention for the prefill stage
- Dense attention for the decoding stage

## News

- **2025.09.29 Release of Sparse Attention Kernels**: With the launch of [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), we are releasing the corresponding token-level sparse attention kernels. These kernels power the model's DeepSeek Sparse Attention (DSA) and achieve up to 640 TFlops during prefilling and 410 TFlops during decoding. We also release a deep-dive blog for our new FP8 sparse decoding kernel. Check it out [here](docs/20250929-hopper-fp8-sparse-deep-dive.md).
- **2025.08.01 Kernels for MHA on SM100**: Thanks to [NVIDIA's PR](https://github.com/deepseek-ai/FlashMLA/pull/76) for MHA forward / backward kernels on SM100!
- **2025.04.22 Deep-Dive Blog**: We'd love to share the technical details behind the new FlashMLA kernel! Check out our deep-dive write-up [here](docs/20250422-new-kernel-deep-dive.md).
- **2025.04.22 Performance Update**: We're excited to announce the new release of Flash MLA, which delivers 5% ~ 15% performance improvement for compute-bound workloads, achieving up to 660 TFlops on NVIDIA H800 SXM5 GPUs. The interface of the new version is fully compatible with the old one. Simply upgrade to the new version for an immediate performance boost! 🚀🚀🚀

## Performance

#### Test & benchmark MLA decoding (Sparse & Dense):

```bash
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_decoding.py
```

The dense MLA decoding kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5 with CUDA 12.8. The token-level sparse MLA decoding kernel (which uses an FP8 KV cache while performing the matrix multiplication in bfloat16) achieves 410 TFLOPS in compute-bound configuration on H800 SXM5 with CUDA 12.8, and achieves up to 350 TFlops on B200 (which is not really optimized yet).

#### Test & benchmark MHA prefill (Dense):

```bash
python tests/test_fmha_sm100.py
```

It achieves up to 1460 TFlops in forward and 1000 TFlops in backward computation on B200, as reported by NVIDIA.

#### Test & benchmark MLA prefill (Sparse):

```bash
python tests/test_flash_mla_sparse_prefill.py
```

It achieves up to 640 TFlops in forward computation on H800 SXM5 with CUDA 12.8, and achieves up to 1450 TFlops on B200, CUDA 12.9.

## Requirements

- SM90 / SM100 (See the support matrix below)
- CUDA 12.8 and above (CUDA 12.9+ is required for SM100 kernels)
- PyTorch 2.0 and above

Support matrix:

| Kernel | GPU Architecture | MLA Mode [2] | KVCache Format |
| :---: | :---: | :---: | :---: |
| Dense Decoding | SM90 | MQA | BF16 |
| Sparse Decoding | SM90 & SM100 | MQA | FP8 [1] |
| Dense Prefill | SM100 | MHA |  |
| Sparse Prefill | SM90 & SM100 | MQA |  |

[1]: For more details on using FP8 KV cache, see documents below.

[2]: Here "MLA Mode" refers to the mode used for MLA calculation. MQA stands for Multi-Query Attention mode (i.e. `head_dim_k` =  576 with `head_dim_v` = 512), while MHA stands for Multi-Head Attention mode (i.e. `head_dim_k` = 192 / 128 with `head_dim_v` = 128). For a detailed explanation of these modes, please refer to the appendix of [DeepSeek V3.2's Paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).

## Installation

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
```

## Usage

### MLA Decoding

To use the MLA decoding kernels, call get_mla_metadata once before the decoding loop to get the tile scheduler metadata. Then, call flash_mla_with_kvcache in each decoding step. For example:

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,
    h_kv,
    h_q,
    is_fp8,
    topk,
)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal, is_fp8_kvcache, indices,
    )
    ...
```

Where

- `s_q` is the number of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1.
- `h_kv` is the number of key-value heads.
- `h_q` is the number of query heads.

**FP8 KV Cache:**
If `is_fp8_kvcache` is set to `True`, the kernel reads the KV cache in the "FP8 with scale" format (described below). It dequantizes the cache to bfloat16 and performs attention computation in bfloat16. The output is also in bfloat16.

In the "FP8 with scale" format, each token's KV cache is 656 Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512 `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values. The first `float32` is the scale for the first 128 `float8_e4m3` values, the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This part is not quantized for accuracy.

See `tests/quant.py` for quantization and dequantization details.

**Sparse Attention (`indices` tensor):**
The `indices` tensor (if provided) enables token-level sparse attention by instructing the kernel to compute attention only for specified tokens.

-   **Shape:** `indices` should be a 3D tensor of shape `(batch_size, seq_len_q, topk)`.
-   **Format:** `indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * page_block_size + (the offset of token t within the page block)`, where `t` is the k-th token for the j-th query sequence in the i-th batch. Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not require the `block_table` parameter.
-   **Invalid entries:** Set invalid indices to `-1`.

**Return Values:**
The kernel returns `(out, lse)`, where:
-   `out` is the attention result.
-   `lse` is the log-sum-exp value of the attention scores for each query head.

See `tests/test_flash_mla_decoding.py` for a complete example.

### Sparse MLA Prefill

For the sparse MLA prefill kernel, call `flash_mla_sparse_fwd` directly with the following parameters:
-   `q`: Query tensor of shape `[s_q, h_q, d_qk]`
-   `kv`: Key-Value tensor of shape `[s_kv, h_kv, d_qk]`
-   `indices`: Indices tensor of shape `[s_q, h_kv, topk]`
-   `sm_scale`: A scalar value

**Note on batching:** This kernel does not support a batch dimension. For multi-batch inference, reshape the input tensors and adjust the `indices` parameter to simulate batch processing.

**Invalid indices:** Set invalid entries in `indices` to `-1` or any number `>= s_kv`.

**Return Values and Equivalent PyTorch Code:**
The kernel returns `(out, max_logits, lse)`. This is equivalent to the following PyTorch operations:

```python
Q: [s_q, h_q, d_qk], bfloat16
kv: [s_kv, h_kv, d_qk], bfloat16
indices: [s_q, h_kv, topk], int32

kv = kv.squeeze(1)  # [s_kv, d_qk], h_kv must be 1
indices = indices.squeeze(1)    # [s_q, topk]
focused_kv = kv[indices]    # For the i-th sequence (s_q), the corresponding KV tokens are selected from the KV cache based on indices[i, :]. This operation results in a tensor of shape [s_q, topk, d_qk].

P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * math.log2(math.e)    # [s_q, h_q, topk]
max_logits = P.max(dim=-1) # [s_q, h_q]
lse = log2sumexp2(P, dim=-1, base=2)   # [s_q, h_q]，"log2sumexp2" means that the exponentiation and logarithm are base-2
S = exp2(P - lse)      # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

See `tests/test_flash_mla_prefill.py` for a complete example.

### Dense MHA Prefill

This kernel implements the standard dense Multi-Head Attention (MHA) forward and backward operations. It can be called using:
-   `flash_attn_varlen_func`
-   `flash_attn_varlen_qkvpacked_func`
-   `flash_attn_varlen_kvpacked_func`

The usage is similar to the `flash_attn` package. See `tests/test_fmha_sm100.py` for a complete example.

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Community Support

### MetaX
For MetaX GPUs, visit the official website: [MetaX](https://www.metax-tech.com).

The corresponding FlashMLA version can be found at: [MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA)


### Moore Threads
For the Moore Threads GPU, visit the official website: [Moore Threads](https://www.mthreads.com/).

The corresponding FlashMLA version is available on GitHub: [MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA).


### Hygon DCU
For the Hygon DCU, visit the official website: [Hygon Developer](https://developer.sourcefind.cn/).

The corresponding FlashMLA version is available here: [OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention).


### Intellifusion
For the Intellifusion NNP, visit the official website: [Intellifusion](https://www.intellif.com).

The corresponding FlashMLA version is available on Gitee: [Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py).


### Iluvatar Corex
For Iluvatar Corex GPUs, visit the official website: [Iluvatar Corex](https://www.iluvatar.com).

The corresponding FlashMLA version is available on GitHub: [Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla)


### AMD Instinct
For AMD Instinct GPUs, visit the official website: [AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html).

The corresponding FlashMLA version can be found at: [AITER/MLA](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient Multi-head Latent Attention Kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
