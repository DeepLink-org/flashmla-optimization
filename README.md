# FlashMLA Active-Head Optimization for vLLM

这是一个基于 DeepSeek FlashMLA 的实验性优化分支，面向 H200（SM90）上的
GLM-5.2-FP8 和 DeepSeek-V4 sparse-decode 生产负载。它保留完整上游 Git 历史，
从 FlashMLA `a8f794d1251cbfd88a5011445dd5582289c727e4` 开始，将本轮优化按功能拆成
独立提交，便于审查、回退、二分和后续向新版本移植。

> 本仓库不是 DeepSeek 官方最新 `main`。代码与验证结论只对应下文冻结的版本、
> ABI、模型形状和 H200 环境；不要把当前二进制直接替换进任意 FlashMLA/vLLM 版本。

## 背景

真实 vLLM profile 中，sparse decode 的两个 FlashMLA CUDA kernel 占比较高：

- `mla_sparse_attention_splitkv`
- `mla_sparse_attention_combine`

GLM-5.2 TP8 和 DeepSeek-V4 TP8 传给 Hopper kernel 的 query 都 padding 到 64 heads，
但每个 rank 实际只消费 8/16 heads。上游 split 主体仍需保留 64-row WGMMA tile，
而它的 epilogue 以及 combine 会为 padding head 做没有消费者的工作。

本分支的最终生产策略不改变 QK、PV、softmax 或真实 head 的归约顺序，主要做三件事：

1. 将 TP-local `active_h_q` 显式传入 sparse-decode metadata；
2. combine 只启动并处理真实 heads，并为 `num_splits <= 32` 提供 exact fast path；
3. splitkv 保留 64-row WGMMA 主体，但跳过 padding heads 的 FP32 scaling 和
   shared-memory epilogue store。

第一层提交还保留了早期 workload-aware SM partition 实验接口，方便历史复现；正式
candidate 使用 `active-combine-only`，不启用改变 split 分组的 `dynamic` 策略。

## 冻结基线与适用范围

| 项目 | 固定值 |
| --- | --- |
| FlashMLA 基线 | `a8f794d1251cbfd88a5011445dd5582289c727e4` |
| CUTLASS 子模块 | `147f5673d0c1c3dcf66f78d677fd647e4a020219` |
| vLLM | `0.27.1`，commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| GPU | NVIDIA H200 / SM90 |
| CUDA | 12.8，构建时禁用 SM100 |
| GLM-5.2 TP8 | padded heads 64，active heads 8，`d_qk=576`，`d_v=512` |
| DeepSeek-V4 TP8 | padded heads 64，active heads 16，`d_qk=d_v=512` |

Hopper `h_q=64` sparse FP8 decode 是已验证边界；`h_q=128` 保持上游模板。

## 已验证收益

GLM-5.2-FP8、单机 8×H200、TP8、baseline/candidate 均开启相同 CUDA Graph：

| 场景 | 三轮请求级 E2E 收益 | 中位收益 |
| --- | --- | ---: |
| 1k 输入 / 32k 输出 | `+2.899% / +3.597% / +3.670%` | **+3.597%** |
| 32k 输入 / 1k 输出 | `+4.389% / +4.560% / +4.447%` | **+4.447%** |
| 4k 输入 / 4k 输出 | `+2.332% / +2.332% / +2.732%` | **+2.332%** |
| 8k 输入 / 8k 输出 | `+2.540% / +3.313% / +3.584%` | **+3.313%** |

DeepSeek-V4 的 kernel 与 steady-state ITL 均有正收益；其带 DSpark 的短样本 makespan
会被 speculative acceptance、输出分叉和 KV/prefix cache 状态显著干扰，不能只看
一次请求总时长。完整边界和解释见 [验证结果](docs/VALIDATION.md)。

## 提交结构

每项修改都是独立提交：

```text
a8f794d  upstream pinned baseline
912c4ab  sparse-decode workload / SM partition policy plumbing
2b9baaa  combine only processes TP-local active heads
39eb8ff  exact fast combine path for num_splits <= 32
c725b63  splitkv skips padded-head epilogue stores
8f61824  offline build uses the pinned CUTLASS checkout
```

每个提交的目的、开关和安全边界见 [Patch 栈说明](docs/PATCH_STACK.md)。

## 获取代码

必须递归检出固定的 CUTLASS 子模块：

```bash
git clone --recursive <your-github-url> flashmla-active-head-optimization
cd flashmla-active-head-optimization
git submodule status
```

若第一次 clone 没有加 `--recursive`：

```bash
git submodule update --init --recursive
```

## 构建扩展

推荐在目标 vLLM 镜像内构建，以保证 PyTorch、CUDA、Python ABI 与运行镜像一致：

```bash
CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include/cccl \
FLASH_MLA_DISABLE_SM100=1 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
python3 setup.py build_ext --inplace

sha256sum flash_mla/_flashmla_C*.so
```

如果目标环境需要 SM100，不要沿用上述开关和本仓库的 H200 验证结论；应重新构建并
完成精度、kernel benchmark 和 E2E 门禁。

## 构建 vLLM 派生镜像

仓库提供单个 multi-stage Dockerfile：builder 使用目标 vLLM 镜像中的完整工具链
编译扩展，runtime stage 只替换 vLLM 的 FlashMLA `.so`。

```bash
docker build \
  --build-arg BASE_IMAGE='registry.h.pjlab.org.cn/ailab-pj-bw_gpu/yangxiaolei@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2' \
  -f docker/Dockerfile.vllm \
  -t vllm-v0271-flashmla-active-head:local \
  .
```

这个 Dockerfile 的目标路径是已验证镜像中的：

```text
/usr/local/lib/python3.12/dist-packages/vllm/_flashmla_C.abi3.so
```

其他 vLLM 镜像必须先确认真实模块路径和 ABI，必要时通过
`--build-arg VLLM_FLASHMLA_PATH=...` 覆盖。不要只凭文件名相同就替换。

## 运行时开关

GLM-5.2-FP8、TP8：

```bash
export WS58_FLASHMLA_POLICY=active-combine-only
export WS58_FLASHMLA_ACTIVE_HEADS=8
export WS58_FLASHMLA_FAST_SMALL_SPLITS=1
```

DeepSeek-V4、TP8：

```bash
export WS58_FLASHMLA_POLICY=active-combine-only
export WS58_FLASHMLA_ACTIVE_HEADS=16
export WS58_FLASHMLA_FAST_SMALL_SPLITS=1
```

同一重编译 binary 内的上游调度/实现基线：

```bash
export WS58_FLASHMLA_POLICY=baseline
export WS58_FLASHMLA_FAST_SMALL_SPLITS=0
```

注意：

- `WS58_FLASHMLA_ACTIVE_HEADS` 必须不大于 padded `h_q` 且是 8 的倍数，否则
  fail-closed 回退到完整 `h_q`；
- `WS58_FLASHMLA_POLICY=dynamic` 会改变 SM partition 数，只用于旧实验复现，不是
  最终生产策略；
- 正式 E2E baseline 应优先使用未经修改的生产镜像；同 binary 的 `baseline` policy
  主要用于隔离调度策略和 CUDA Graph microbenchmark；
- 环境变量必须在 vLLM worker 启动前注入。

## 正确性和性能验证

最低验收顺序：

1. 使用固定 seed、真实 KV layout 和真实 shape 对比 active output/LSE hash；
2. 以 FP64 oracle 计算模型可见 active output 的 MAE，并要求 candidate 不劣于
   production BF16 baseline；
3. baseline/candidate 使用相同 CUDA Graph，做多轮交错 kernel benchmark；
4. 启动真实 TP8 vLLM，按 `B1-C1-C2-B2-B3-C3` 交错跑三轮；
5. 同时审计镜像 ID、扩展 SHA256、JIT、GPU 独占、prompt/token、TTFT、TPOT、ITL；
6. speculative decode 模型额外记录 acceptance 和输出 hash。

GLM FP64 oracle 工具见 `campaign/bench_glm_flashmla_fp64_oracle.py`。它要求在包含
目标 vLLM 接口和 H200 GPU 的运行环境中执行。

## 仓库结构

```text
csrc/                         FlashMLA CUDA/C++ 源码
flash_mla/                    Python 接口
campaign/                     本轮新增的验证工具
docker/Dockerfile.vllm        构建并替换 vLLM 扩展
docs/PATCH_STACK.md           每个提交的修改和回退方法
docs/VALIDATION.md            精度、性能结果与解释边界
```

## 上游、许可证与版本策略

- 上游项目：[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- 本分支基线：[a8f794d](https://github.com/deepseek-ai/FlashMLA/commit/a8f794d1251cbfd88a5011445dd5582289c727e4)
- 许可证：沿用上游 [MIT License](LICENSE)

本地 remote 名称 `upstream` 指向 DeepSeek 官方仓库；在创建自己的 GitHub 仓库后，
请把新地址添加为 `origin`，不要覆盖 `upstream`：

```bash
git remote add origin <your-github-url>
git push -u origin main --tags
```

---

## Upstream FlashMLA documentation

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
