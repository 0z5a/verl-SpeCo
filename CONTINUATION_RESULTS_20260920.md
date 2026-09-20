# L20 续做结果 — 2026-09-20

提交者：0z5a。先核查并复用了机器上的既有分支、模型和测试记录；没有将历史完成的 EAGLE3 / DFlash 20 步闭环重新冒充本轮新增结果。已检查已提交代码，并基于上游 main `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3` rebase。附件 skill 仅用于正确性、筛选和独立验证流程。

## 范围与交付

| 项目 | 本轮结果 | 分支 / 提交 |
|---|---|---|
| C1 frozen P-EAGLE | 实际训练 checkpoint → 转换 → vLLM 生成 → 32 次缓存 logits 对照通过；25 张量完全一致，argmax 全同，最大误差 0.00390625 | `0z5a/speco-peagle-frozen-serving` / `fc273cf` |
| C2 sequence partition | 两卡独立 A0/P0/P1/A1 完整训练进程，各 6 更新、2 次 checkpoint、干净退出；44 项回归通过 | `0z5a/speco-partition-e2e` / `9be33cc` |
| C3 bubble-time workers | 未重复实现：维护者已有初版并明确要求优先 VeOmni，待其提交后 review | [维护者说明](https://github.com/verl-project/verl-SpeCo/issues/7#issuecomment-5742262814) |
| C4 VeOmni drafter | 新增独立进程 dense P-EAGLE adapter；两卡参数、梯度、裁剪、AdamW moments 对照通过；6 步训练、保存和 6→8 恢复通过；61 passed / 1 dependency skip | `0z5a/speco-veomni-drafter` / `6df26e4` |
| C5 原 PR10 public loader | 仍未通过：vLLM 0.29.0 重复添加 model. 前缀；原 PR 未给配套幂等处理提交，已请求其 SHA | 原 PR head `b258ec517977a01df722909789da42746c39f28f` |

C4 review 修复了 P-EAGLE 导出时未取内部 draft_model，以及恢复 checkpoint 时重新覆盖已训练 embedding 的两个问题。新路径没有引入 Any、getattr 或宽泛异常捕获。各分支八项仓库 sanity 检查通过；并不声称全仓测试全部通过。

## 速度比较

速度提升 = baseline / candidate − 1；多轮使用耗时几何均值。单位和工作量保持各行一致。

| 测试 | baseline | candidate | 观测速度提升 | 判定 |
|---|---:|---:|---:|---|
| C2 两卡完整小模型训练，6 更新，A0/P0 | 18.587 s | 23.261 s | −20.09% | 一组观测 |
| C2 同上，A1/P1 | 23.780 s | 23.044 s | +3.19% | 一组观测 |
| C2 几何均值 | 21.024 s | 23.152 s | −9.19% | baseline 波动 27.94%，无加速结论 |
| C4 两卡完整小模型训练，6 更新，A0/P0 | 96.462 s | 21.617 s | +346.23% | baseline 退出延迟，性能比较无效 |
| C4 同上，A1/P1 | 18.828 s | 19.585 s | −3.87% | 一组观测 |
| C4 几何均值 | 42.617 s | 20.576 s | +107.12% | baseline 波动 5.12 倍，不认定加速 |
| C1 2 prompts × 16 tokens | 32 tokens | 32 identical tokens | N/A | 数值验证含 hook，无受控计时 |
| 历史 C2 真实 target，256 tokens 单 optimizer step | 322.22 ms | 564.92 ms | −42.96% | 不是完整训练流程 |
| 历史 C2 真实 target，1024 tokens 单 optimizer step | 551.88 ms | 886.93 ms | −37.78% | 峰值显存 21.05→16.28 GiB，下降 22.69% |

C1/C2/C4 本轮完整流程使用 tiny 全词表 P-EAGLE，不能替代真实 4B 在线 RL、有效 reward、TP2/TP4 frozen serving 或 CUDA Graph 验证。恢复测试证明 checkpoint 生命周期和 optimizer step 连续性，不证明 COD RNG 和数据游标等同于不中断运行。历史真实 EAGLE3、DFlash 各 20 步训练—发布—后续 rollout 通过，但 actor gradient / reward 为零；详见 `TEST_RESULTS_L20.md`。

## 原始记录与复现

每个实现分支的 `TEST_RESULTS.md`、`experiment/l20/` 和 `evidence/l20-20260920/` 包含命令、版本、原始日志、比较脚本及退出码。C1 保留 forward 输入输出捕获，以检查 padding、masked tokens 和拒绝后 KV 重用。vLLM skip-tokenizer chat-template 预热警告不影响 token-ID 生成。原始日志未清洗。

本轮提交保留在本机，尚未 push 或创建 PR。C3/C5 及上述未覆盖的真实 RL 范围仍未完成，不将五项标为全部 E2E 通过。

最终再次 fetch GitHub 时网络连接超时；rebase 对照的是本轮此前成功获取的 main SHA，未声称最后一次 fetch 成功。

## 已完成模型清理

远端删除 66 个本任务专属模型 / optimizer 权重文件，释放 8,080,992,787 bytes（7.526 GiB）；本地删除 3 个 tiny draft 权重，释放 2,271,968 bytes。每个文件的路径、大小和 SHA-256 已记录在 `completed-model-cleanup.json` / `local-completed-model-cleanup.json`。保留配置、原始日志和 forward 捕获；未触碰共享缓存或其他任务模型。C5 后续拿到依赖后需重新下载 target。

## 再次续做：Graph 与 public loader 候选依赖

C1 新提交 `cd4c382`，最终记录提交 `4593dd0`：重新生成原始 tiny fixture，完成单卡 BF16 CUDA Graph 的 baseline / P-EAGLE 两轮生成。两轮输出全部一致，并与此前 eager 输出逐 token 一致。实际观测 baseline 80 次 replay、speculative engine 272 次 replay；25 个逻辑权重张量完全一致。角色计数包括 target 和 PiecewiseBackend，未把后者逐个映射到 draft layer；没有声称 graph 下完成独立的 draft logits oracle。

| 本轮对照 | 原路径 | 新路径 | 速度提升 | 正确性结果 |
|---|---|---|---|---|
| C1 Graph 两轮生成 | target-only，64 tokens | P-EAGLE，64 tokens | N/A：带观测 hook、共享 GPU，未受控计时 | 全部 token 一致，两个进程 exit 0 |
| C5 prefixed public-loader | vLLM 0.29.0，重复前缀报错 | 本地一行幂等前缀补丁，连续加载两次 | N/A：基线功能失败，工作量不可比 | fc.weight 精确更新为 0.5 倍，后续生成通过 |

C5 补丁只将 `elif "lm_head" not in name` 改为额外检查 `not name.startswith("model.")`。测试使用 tiny 全词表 P-EAGLE checkpoint 和 vLLM 的 Eagle3LlamaForCausalLM，运行完整参数名集合的 public load_weights，包含明确变更的 fc.weight。它是本地候选依赖的 GPU 验证，不是原作者配套提交，也不是原 PR10 的 IPC、真实 EAGLE3/DFlash RL、DFlash fused-KV 或在线 CUDA Graph 更新验证。记录见 `evidence/l20-20260920/public-loader-*`；补丁和复现脚本在 `experiment/l20/`。

执行前后安装文件 SHA-256 均为 `5be7ee5513499b982d08803ac138926954377c858ffc2f2ebfe123e65556a61a`，确认测试脚本已恢复容器中的 vLLM 源码。当前 vLLM main `751f6807d9cb3de50c27a5f27188c4fb04fe0e2b` 的该 loader 仍无前缀幂等 guard；SpeCo main 仍为原固定 SHA，C3 未发现发布的对应实现。

TP2 target-only baseline 在 FlashAttention 内停滞。已尝试 NCCL 替代 custom all-reduce、V1 替代 V2，以及 spawn 替代 fork；未把卡住归因于 P-EAGLE，也未申报 TP2 通过。一次独立启动失败明确是显存预留门槛不足。原始日志与 Python worker stack 已保存在 C1 分支。机器多卡被其他工作占用，已请求可用于完整 4B 在线 E2E 的 GPU 编号或预留时段，未停止其他任务。

本轮完成后再次清理 4 个重新生成的 tiny 模型权重，释放 2,271,968 bytes；路径、大小和 SHA-256 见 `evidence/l20-20260920/continuation-model-cleanup.json`。所有本轮残留 engine / worker 已停止，配置和证据保留。Graph 与 loader 的结果已提交；TP2、C3、C4 的精确 RNG/数据位置恢复与真实在线发布、原 C5 的双算法完整 RL 验证仍未全部完成。

## C5 环境调整

用户确认 C5 使用 L20 主机原生 vLLM。已锁定 `agent_use` 的 vLLM 0.18.0 / PyTorch 2.10.0+cu128，并适配该版本的 fixture / worker 接口。首次原生对照因导入阶段文件页等待而超时；下方续测已完成 loader 对照，并发现及定位 parallel-drafting 派生缓存问题。原生 registry 仍缺少 DFlash。详见 `C5_NATIVE_RESULTS.md`。原生共享安装未修改，测试权重已清理。


## C5 原生环境续测

原生 0.18 的 baseline 已复现具体 `KeyError: model.embed_tokens.weight`；独立一行前缀候选完成两次 public load、fc 精确变更和后续生成。新增独立 cold-B oracle 发现 parallel-drafting proposer 的 mask hidden cache 未刷新：仅前缀候选的 draft logits 最大误差为 0.0556640625，尽管 target hash 和最终 tokens 均相同。测试侧原地刷新缓存后 1,280 logits 精确一致，地址保持不变。这是诊断结果，不是完整 PR10 / 双算法 RL 通过。

| 对照 | 修改前 | 修改后 | 速度提升 |
|---|---|---|---|
| 原生 public loader | 重复前缀 KeyError | 前缀候选连续加载通过 | N/A，基线失败 |
| hot-B 对 cold-B draft logits | 仅前缀候选误差 0.0556640625 | 测试侧刷新缓存后误差 0 | N/A，正确性专项 |
| 原生 EAGLE3 / DFlash 完整 RL | 未完成 | 未完成 | N/A |

本轮 5 个模型权重文件已清理，释放 2,786,856 bytes；报告及复现见 `C5_NATIVE_RESULTS.md`，原始日志和 logits 在 `evidence/l20-20260920/native/resume-0910/`。原生共享安装没有修改。DFlash 在现有 0.18 缺少实现，另建原生环境的选择已询问用户。

## Latest native environment, C2 fix and C3 coordination

- Native vLLM 0.29.0 is installed separately with PyTorch 2.13.0+cu130 and
  Transformers 5.17.0. The official wheel hash and dependency check pass;
  `NATIVE_VLLM_SETUP_20260920.md` records the environment.
- C2 commit `ae8b5a2` removes vocabulary-head/KL work for context-only positions.
  The 1,024-token Qwen3-4B optimizer-step comparison is +19.44% versus the old
  partition, with peak allocation 16.28 → 14.23 GiB. Relative to old flat it is
  still −22.12% speed and −32.40% memory. Full standalone timing is inconclusive;
  all four six-step/save lifecycle runs pass. No full RL speedup is claimed.
- The C3 coordination comment was posted as 0z5a:
  https://github.com/verl-project/verl-SpeCo/issues/7#issuecomment-5746767473
- C2 completed-task weights were cleaned (23 files, 8,051,385,896 bytes). The
  small copied checkpoint remains only while C1 is actively validating it.
