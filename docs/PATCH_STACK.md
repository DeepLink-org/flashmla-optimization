# Patch 栈与提交边界

本仓库保留 DeepSeek FlashMLA 上游历史，`main` 从固定提交
`a8f794d1251cbfd88a5011445dd5582289c727e4` 向前追加本轮修改。以下顺序也是构建
最终 candidate 的顺序。

## 0. 上游固定基线：`a8f794d`

该提交是 vLLM 0.27.1 生产集成所对应的 ABI-stable FlashMLA 源码。本仓库没有基于
2026-08-24 的官方最新 `main` 开发；上游后续已经修改 combine grid 和 split bucket，
需要独立移植和重新验证，不能直接套用本仓库结论。

## 1. `912c4ab`：workload-aware SM partition policy

修改文件：`csrc/api/sparse_decode.h`。

- 把 batch、`s_q`、`h_q`、`d_qk`、top-k 和 extra KV 等信息整理为 workload；
- 识别 GLM、DS-V4 SWA/C4A/C128A sparse decode signature；
- 提供 `baseline`、`dynamic`、`fixed` 等运行时 policy；
- 无法识别或参数非法时回退上游 partition 数；
- 输出有限次数的 `WS58_FLASHMLA_DYNAMIC_HIT` 诊断日志。

这是早期 C04/partition-cap 实验的基础设施。最终 active-head candidate 使用
`active-combine-only`，保持上游 SM partition 数。

## 2. `2b9baaa`：active-head combine

修改文件：`csrc/api/sparse_decode.h`。

- 从 `WS58_FLASHMLA_ACTIVE_HEADS` 读取 TP-local 真实 head 数；
- 只有 `dynamic` 或 `active-combine-only` policy 才允许启用；
- 要求 `0 < active_h_q <= h_q` 且是 8 的倍数；
- combine grid 使用 `active_h_q`，不再处理 padding heads；
- split kernel 的 padded query ABI 在这一提交中仍保持不变。

GLM-5.2 TP8 使用 8；DeepSeek-V4 TP8 使用 16。

## 3. `39eb8ff`：exact small-split combine

修改文件：`csrc/smxx/decode/combine/combine.cu`。

- 用模板参数保留 upstream combine 和 fast combine 两套实现；
- 当 `WS58_FLASHMLA_FAST_SMALL_SPLITS=1` 且真实 `num_splits <= 32` 时，只归约有效
  LSE lanes；
- 避免为不存在的 split 构造 masked `-inf`/zero 工作；
- `num_splits > 32` 自动执行与上游等价的通用路径；
- 不改变真实 split 的累加顺序。

## 4. `c725b63`：active-head split epilogue

修改文件：

- `csrc/api/sparse_decode.h`
- `csrc/params.h`
- `csrc/sm90/decode/sparse_fp8/config.h`
- `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`

这一提交把 `active_h_q` 继续传入 split params。Hopper WGMMA 主体仍计算固定 64 行，
但 epilogue 只为真实 heads 执行 output scaling 和 shared-memory store。

安全边界：

- 只对 `NUM_HEADS == 64` 编译选择 active epilogue；
- `active_h_q >= h_q` 时执行 upstream 模板；
- `NUM_HEADS == 128` 始终执行 upstream 模板；
- 被跳过的是没有模型消费者的 padding output slots。

## 5. `8f61824`：offline source build

修改文件：`setup.py`。

上游 `setup.py` 会在构建过程中执行 `git submodule update`。本提交移除该隐式联网
动作，要求用户在 clone 阶段显式执行：

```bash
git clone --recursive ...
# 或
git submodule update --init --recursive
```

这样 Docker/offline build 的依赖身份由 Git tree 固定，不会在编译中途改变。

## 逐项回退与移植

查看某一项完整 diff：

```bash
git show --stat 2b9baaa
git show 2b9baaa
```

从上游基线逐项构建：

```bash
git switch --detach a8f794d
git switch -c reproduce/active-head
git cherry-pick 912c4ab 2b9baaa 39eb8ff c725b63 8f61824
```

向新版 FlashMLA 移植时必须逐提交 cherry-pick/rebase，并重点检查
`csrc/api/sparse_decode.h`、`csrc/params.h`、sparse splitkv 和 combine 的 ABI/launch
变化。移植成功只代表代码能编译，不能继承本仓库已有的精度和性能结论。
