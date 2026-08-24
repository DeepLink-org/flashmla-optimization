# 验证结果与结论边界

## 固定环境

| 项目 | 值 |
| --- | --- |
| GPU | 单机 8×NVIDIA H200，TP8 |
| driver | `570.133.20` |
| vLLM | `0.27.1`，`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| FlashMLA source | `a8f794d1251cbfd88a5011445dd5582289c727e4` |
| production baseline extension | `c670bc1b91d1a3792d4e6ab477695d65db011778b5becd08283ecc96925cec4b` |
| final candidate extension | `1b5361cc3c146f91c0feeed9e58ecb03e38dc12ee190a711f4e3b597314625a1` |

baseline/candidate 均开启相同 CUDA Graph。正式请求按三轮交错顺序执行，并审计镜像、
扩展哈希、输入输出 token、JIT 和 GPU 独占。

## GLM-5.2-FP8 E2E

| 场景 | R1 | R2 | R3 | 中位 |
| --- | ---: | ---: | ---: | ---: |
| 1k32k | +2.899% | +3.597% | +3.670% | **+3.597%** |
| 32k1k | +4.389% | +4.560% | +4.447% | **+4.447%** |
| 4k4k | +2.332% | +2.332% | +2.732% | **+2.332%** |
| 8k8k | +2.540% | +3.313% | +3.584% | **+3.313%** |

四场景共十二个 paired repeats 全部为正收益。改善主要出现在 decode/steady-state
TPOT、ITL，同时长输入场景也观察到 TTFT 正收益。

## DeepSeek-V4

固定 16k 输入 / 512 输出的 C1/C4/C8/C16 三轮 E2E 中位收益分别为：

| 请求并发 | 中位 E2E 收益 |
| ---: | ---: |
| 1 | +2.855% |
| 4 | +2.565% |
| 8 | +1.720% |
| 16 | +1.162% |

与 GLM 对齐的长场景中，32k1k 三轮为 `+2.948% / +1.042% / +1.144%`，输出 hash
三轮一致；这是当前最干净的 DS-V4 请求级证据。8k8k 的两轮负 makespan 同时发生
DSpark acceptance 明显下降，而 ITL 中位仍改善，因此不能归因成 FlashMLA kernel
回退。

对 speculative decode 模型，必须联合报告：

- 输出 token hash；
- draft acceptance / accepted tokens；
- TTFT、TPOT、ITL；
- prefix/KV cache 是否命中；
- 同服务、同 prompt、同 seed 的执行顺序。

## Kernel 负载边界

GLM 固定 `seq_q=4` transition sweep 中，最大稳定超过 1% 的测试点为 108 query
rows/graph，112 起低于 1%。DS-V4 `seq_q=1` 的相邻边界为 batch 96/104。超过边界
不会改变精度，但 active-head 优化的占比会被更高并发下的其他工作摊薄。

## 精度口径

候选与同源码重编译 baseline 在模型可见 active heads 上做固定 seed、真实 KV layout、
eager/CUDA Graph hash 和逐位一致检查。跨 production binary 的 FP64 oracle 汇总为：

| 对象 | production BF16 baseline MAE | candidate MAE | 变化 | 判定 |
| --- | ---: | ---: | ---: | --- |
| active attention output | 3.761783424e-6 | 3.761699651e-6 | -8.3773e-11 | PASS |
| LSE | 2.197571764e-7 | 3.518673997e-7 | +1.3211e-7 | FAIL |

生产验收以模型可见 BF16 attention output 为严格基准；候选 output 的 FP64 MAE 不劣于
baseline。GLM 和当前 DS-V4 NVIDIA FlashMLA decode 调用均丢弃返回 LSE，不把它送入
后续 Transformer 层。LSE 仍是 split/combine 内部完成 softmax 归一化所必需的中间量，
“调用方丢弃”不代表 kernel 内部可以删除 LSE 计算。

上述边界必须完整保留。若新的消费者开始使用返回 LSE，则当前跨 binary 精度门禁不再
充分，必须把 LSE 也恢复为严格不劣于 baseline。

## 不可外推的结论

- 结果不适用于 FlashMLA 最新 `main`；
- 结果不适用于 B200/SM100；
- 结果不自动适用于非 TP8 或不同 TP-local active head 数；
- 结果不意味着 `dynamic` SM partition policy 已获生产验收；
- 单算子加速不能替代真实 vLLM E2E 验证。
