# L20 验证结果

提交者：0z5a。基线：`18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3`。

**真实 EAGLE3 在线链路已完成 20 步；C1–C5 整体尚未完成。下表只衡量 C2 的完整 drafter optimizer step，不是完整 RL step。C2 是显存与计算时间的交换，不是异步加速。**

## 在线 EAGLE3：20 步结果

基线加 EAGLE3 架构别名补丁；真实 Qwen3-4B/EAGLE3 checkpoint，2×L20、TP2、FSDP2、BF16 actor，真实权重 `load_format=auto`，32-token feature 窗口，异步 compat publish。未启用 C2。

| 核验项 | 实测结果 |
|---|---:|
| 完整运行 | 20/20 步，退出码 0，无 traceback |
| drafter 样本收集 | 40 |
| drafter optimizer | 每步成功，共 20 步 |
| 发布完成 | revision 1–20，每个均双 TP rank 提交，13 weights/rank |
| 后续 rollout | 使用此前已发布 revision；sleep/wake 保留 revision |
| 平均整步时间（含首步） | 27.62 s，单次运行，无速度对照结论 |
| acceptance length | 1.709–2.039 |
| actor 梯度 / reward | 全为 0；不证明 actor 学习或 RL 质量 |

证据：`online-eagle3-final.json`、`online-eagle3-clean-final.log`。最后一次发布发生于训练结束，revision 20 不包含后续训练 rollout；revision 1–19 的后续 rollout 已执行。这是当前主线加 alias 补丁的训练—发布生命周期验证，**不是原 PR #10 的 public-loader E2E，也不是 C2/C1/C3/C4 验证**。

## C2：速度与显存

Qwen3-4B 固定 target；2 层 P-EAGLE、8 depths、完整词表、BF16；1 张 L20。
相同输入、COD 样本计数、optimizer 步数；每个长度按 A/P/P/A 独立启动，每次 3 个预热步、5 个测量步。
时间包括 target-head loss、draft forward、backward、梯度裁剪和 optimizer，不包括模型加载及 target feature 准备。
表中时间为两个独立进程合计 10 个测量步的算术平均，显存为两次启动的最大 allocated 峰值。

| 输入 tokens | 原始主线 ms/step | 2 分区 ms/step | 速度提升¹ | 原始显存 GiB | 分区显存 GiB | 显存降低 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 322.22 | 564.92 | **−42.96%** | 10.25 | 10.32 | −0.68% |
| 1024 | 551.88 | 886.93 | **−37.78%** | 21.05 | 16.28 | **22.69%** |

¹ 速度提升 = `(原始时间 / 分区时间 − 1) × 100%`，负值为变慢。

默认 `peagle_sequence_partitions=1` 保留非分区路径。分区只在需要节省长序列显存时启用。
节点共享；测试卡为物理 GPU 2，其他卡有独立作业。未锁 GPU 时钟。单组四次启动只证明本次实测方向，不给出统计显著性结论。

## 正确性与来源

- 固定 COD 的 CPU FP32 测试：**25 passed**；覆盖单 token、零有效 loss、单/双文档、完整/缩减词表及 1/2/3/20 分区。
- 最终 L20 FP32 测试：**25 passed**；结果保存于 `evidence/l20-20260919/cuda-final-partition.xml`。
- 两卡 FSDP2：两个 rank 的全局 token 归一化梯度和 SGD 更新均通过。每个 rank 验证实际导入路径和文件 hash。
- BF16 实模型性能运行完成；多步 BF16 轨迹存在数值差异，不能由 FP32 对照推导出完整训练质量等价。
- 最终受影响 contract 套件：20 passed / 1 failed；失败是已有 checkpoint-export fixture 缺少 backend，与未修改主线的同名失败一致。
- 主线回归未全绿：初始 331 passed / 9 failed / 1 skipped；补齐部分依赖、禁用 CUDA 后重跑为 351 passed / 18 failed。两次环境条件不同，不能把变化归因于本补丁；完整日志与 JUnit 均保留。
- Ruff、`git diff --check`、修改文件的 `mypy --follow-imports=skip`（默认 Python 3.10 target）通过；本地 NumPy 固定为 2.2.6。跟随全部导入的 mypy 未通过，报告 24 个文件共 97 个错误；不声称完整类型检查通过。新增生产代码未引入 `Any`、`getattr` 或异常捕获。

8 项仓库 sanity 检查通过，详见 `sanity-checks.json`。pre-commit wrapper 因 GitHub git 连接超时未完成，未标记为通过。

最终候选文件 SHA-256：`4571702bed953f3df1f4561f2a0b1b5597248cbeb91d4936b604835e3a3b0ceb`。

Algorithm 1 分区归属依据 [NeMo AutoModel 固定源码](https://github.com/NVIDIA-NeMo/Automodel/blob/e2c47c5bcb3e6e0224adace569a1ded7872c002f/nemo_automodel/components/speculative/eagle/peagle_data.py)。本实现用非重入 checkpoint 释放分区激活，保持原有 wrapped forward/backward 边界。

## C1–C5 范围

| 项目 | 本次状态 |
|---|---|
| C1 P-EAGLE serving | 未实现；主线仍明确拒绝该 rollout 算法，未删除 guard 假装支持。 |
| C2 sequence partition | 开发版本；FP32 正确性、两卡 FSDP2 和上述实际训练步 A/B 已执行。尚无完整 RL E2E。 |
| C3 idle worker | 未实现；现有 scheduler 只执行 sync，idle enum 本身不是执行能力。 |
| C4 VeOmni drafter | 未实现；当前 drafter 仍经 FSDP 包装，不能把 VeOmni actor 当成 drafter 验证。 |
| C5 PR #10 | 真实 Qwen3-4B + EAGLE3 CUDA 推理生成 16 token；public loader 接受 `fc.weight`，拒绝 `model.fc.weight`，参数保持不变。缺少 #10 要求的幂等前缀依赖已复现，见 `public-loader.json`。不是原 PR 的训练—发布闭环通过。 |

## 复现

模型已固定：

| 模型 | Revision |
|---|---|
| Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| AngelSlim/Qwen3-4B_eagle3 | `fd331e59626c8e95c392381a16ee59d518727fbb` |
| z-lab/Qwen3-4B-DFlash-b16 | `b74e3a329c4d963783143b1e970d95b002be72bd` |

性能环境：vLLM 0.29.0 镜像、PyTorch 2.13.0+cu130、transformers 5.16.1。
在线循环另用 transformers 5.10.4，符合 verl 0.9.0 与 vLLM 0.29.0 的依赖交集。

```bash
python -m pytest -q tests/integration/test_peagle_partition.py
SPECO_TEST_DEVICE=cuda python -m pytest -q tests/integration/test_peagle_partition.py
SPECO_SOURCE=/experiment/variants/patch python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29587 \
  experiment/l20/check_fsdp_partition.py
bash experiment/l20/run_training_ab.sh
```

A/B harness 需要分别把固定基线与候选放在 `/experiment` 和 `/experiment/variants/patch`，模型位于 `/experiment/models/target`。所有原始结果均保存在本任务的 `evidence/l20-20260919/`；`final/` 为最终源码的独立进程测量，`screen/` 不混入最终表格。

## 在线链路排障

主线对真实 AngelSlim checkpoint 报 `Architecture Eagle3LlamaForCausalLM not supported`。
补丁只把该名称加入已有 EAGLE3 alias/mapping；checkpoint 的 fc、attention、MLP、词表映射 tensor 名称与形状匹配现有 trainer，4 项配置测试通过。
该补丁与 C2 分区分别提交；在线实验只使用 alias 补丁，不启用 C2。

Ray worker 最终执行路径为 `/experiment/variants/eagle-alias/verl_speco/`。
先前仅设置 PYTHONPATH 的尝试仍从原目录导入，不算候选验证；复现脚本现从 SPECO_SOURCE 启动。
`online-eagle3-alias-cwd.log` 证明 drafter checkpoint 加载已通过，随后 actor 分桶权重发布失败：
`lm_head.weight` 的 tied embedding 不在同一 bucket，vLLM 0.29 的 loader 拒绝该次更新。

代码提交者为 0z5a；C2 commit `04bf864`，EAGLE3 alias commit `bc00ecc`。
2026-09-19 GitHub API 返回的 main SHA 仍为本报告固定基线；git fetch 连接超时，不声称 fetch 成功。

### 在线运行的进一步结果

- FP32 actor 的 10 GiB bucket 仍会拆分 tied weights；20 GiB bucket 在 L20 上 OOM。
- BF16 actor + 10 GiB bucket + rollout memory utilization 0.25 可通过 actor 权重同步。
- FlashAttention 2.8.3 源码构建完成，FP16/BF16 CUDA 前向、反向、padding 均通过；版本和扩展 hash 见 `online-environment.json`。
- 默认 512-token feature 窗口大于 64-token response，导致零训练；该运行不算闭环。修正为 32-token 窗口与 remove-padding 后，首步收集 2 个样本，完成 1 个 drafter optimizer step。
- 随后的异步发布在两个 TP rank 都报 `missing=['lm_head.weight']`。首步 `drafter/published=1` 只表示异步任务已提交，后续 RPC 实际失败；该指标不能单独证明发布成功。详见 `online-first-step.json` 和原始日志。
- 该次运行的 20 步训练—发布—后续 rollout E2E 未通过，未以这组失败实验申报速度收益。

后续诊断发现在线 rollout 的默认 `load_format=dummy` 未加载真实 draft checkpoint，发布时参数清单缺少独立 `lm_head.weight`。独立 V1/V2 实权重双卡 probe 均确认 32k draft head 存在、每卡形状 `[16000, 2560]`，且不共享 target head。此前 dummy 运行不构成真实 checkpoint 的 E2E。最终复现脚本显式设置 `load_format=auto`。

真实 checkpoint 首轮重跑出现 OOM，进程清单定位到本任务旧 VLLM worker 残留；清理后 GPU 2/3 各约 932 MiB 占用，再按相同配置运行。该 OOM 不作为干净环境容量结论。

清理后的最终运行完成 20 步，结论见本文首表；下方历史失败记录不替代最终运行结果。

## DFlash 在线对齐回归

相同 target、TP2、BF16、真实 DFlash checkpoint，20 轮首次运行完成但每轮训练失败：
`input_rows=34, hidden_rows=33, mask_rows=34`。原收集路径把 EAGLE3 的后移 token 窗口用于 block drafter；外层捕获异常后只报告 `no_trainable_batch`，退出码 0 不能说明 E2E 通过。

修复使用算法已有的 block-drafter 分类，令 token、hidden、position 同位置对齐；EAGLE3/P-EAGLE 保持原移位。
36 项窗口回归在原方法上 24 failed / 12 passed，在修复后 36 passed。相关套件 91 passed / 2 failed；两项失败均为已有 worker fixture 缺少 `replica_rank`，在保存的未修改主线结果中同样出现。Ruff、修改文件 mypy、diff 检查通过。

DFlash 完整日志已取回：20 次 optimizer 更新、40 次 rank 提交，revision 1–19 均在后续 wake-up 恢复。最终 revision 20 没有后续 rollout。平均整步 60.10 s；actor 梯度/reward 仍为 0。训练结束后 DataLoader worker 64271 被 SIGKILL，外层退出码仍为 0，严格 clean-exit 检查未通过。随后以 `data.dataloader_num_workers=0` 独立重跑完成：退出码 0，无 traceback，严格检查通过；40 样本、20 optimizer 更新、40 rank 提交，revision 1–19 恢复。平均整步 83.19 s，acceptance 2.090–2.758，actor 梯度仍为 0。原始日志的第 4 步被 Ray 拆行；摘要器重接 actor 前缀后的续行，原日志保持不变。P-EAGLE runtime 的源码兼容矩阵见 `experiment/l20/PEAGLE_COMPATIBILITY.md`；尚无 logits parity。

| DFlash TP2 对照 | 原主线 | 对齐修复（最终独立重跑） | 速度提升 |
|---|---:|---:|---|
| 完成 RL 外循环 | 20 | 20 | 不适用 |
| 实际 drafter optimizer 更新 | 0 | 20 | 不适用：基线训练失败 |
| TP rank 发布确认 | 0 | 40 | 不适用 |
| 平均整步耗时（含首步） | 31.65 s | 83.19 s | 不可比较：有效工作量不同 |
| 干净退出检查 | 训练 traceback，失败 | 无 traceback，通过 | 不适用 |

最终证据：`online-dflash-clean.log`、`online-dflash-clean.json`、`online-dflash-clean.exit`。最终运行关闭数据加载子进程；共享节点且非配对性能实验，不能比较两次修复运行的耗时来宣称加速。

历史完整阶段审计：`evidence/l20-20260919/online-dflash-aligned-audit.json`。该运行证明对齐修复恢复了训练和发布，不证明原 PR #10 的 outer public-loader 路径、CUDA Graph 或 RL 质量。

## 模型清理

按要求删除已完成在线验证的 EAGLE3 权重 `model.safetensors`，释放 436,899,680 bytes；配置、revision 和测试证据保留。DFlash 干净重跑通过后删除其 `model.safetensors`，再释放 1,074,860,568 bytes；配置和证据保留。Qwen3-4B target 暂留用于后续验证。分支已改为 `0z5a/speco-l20-validation`。
