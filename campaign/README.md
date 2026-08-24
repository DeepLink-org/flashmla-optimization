# Campaign tools

本目录只放与 GLM/DS-V4 active-head 优化直接相关的验证工具。上游通用 FlashMLA
tests/benchmark 仍保留在原目录中。

`bench_glm_flashmla_fp64_oracle.py` 使用 GLM-5.2 TP8 的生产 sparse-decode shape：

- padded heads 64，active heads 8；
- `d_qk=576`，`d_v=512`；
- top-k 2048；
- 每 token 656-byte FP8 KV record；
- 固定 seed，同时检查 eager/CUDA Graph 一致性和 FP64 oracle MAE。

脚本通过 vLLM 的 FlashMLA wrapper 调用扩展，因此必须在匹配的 vLLM 0.27.1/H200
环境中运行。例如：

```bash
python3 campaign/bench_glm_flashmla_fp64_oracle.py \
  --policy baseline \
  --output /tmp/glm-baseline.json

python3 campaign/bench_glm_flashmla_fp64_oracle.py \
  --policy fast-combine \
  --output /tmp/glm-candidate.json
```

不要在不同 binary、不同 CUDA Graph 配置或不同输入 seed 之间直接比较时延/哈希。
