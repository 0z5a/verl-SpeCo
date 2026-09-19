# L20 续做结果 — 2026-09-20

提交者：0z5a。先核查并复用了机器上的既有分支、模型和测试记录；没有将历史完成的 EAGLE3 / DFlash 20 步闭环重新冒充本轮新增结果。已检查已提交代码，并基于上游 main `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3` rebase。附件 skill 仅用于正确性、筛选和独立验证流程。

## 范围与交付

| 项目 | 本轮结果 | 分支 / 提交 |
|---|---|---|
| C1 frozen P-EAGLE | 实际训练 checkpoint → 转换 → vLLM 生成 → 32 次缓存 logits 对照通过；25 张量完全一致，argmax 全同，最大误差 0.00390625 | `codex/speco-peagle-frozen-serving` / `fc273cf` |
| C2 sequence partition | 两卡独立 A0/P0/P1/A1 完整训练进程，各 6 更新、2 次 checkpoint、干净退出；44 项回归通过 | `codex/speco-partition-e2e` / `9be33cc` |
| C3 bubble-time workers | 未重复实现：维护者已有初版并明确要求优先 VeOmni，待其提交后 review | [维护者说明](https://github.com/verl-project/verl-SpeCo/issues/7#issuecomment-5742262814) |
| C4 VeOmni drafter | 新增独立进程 dense P-EAGLE adapter；两卡参数、梯度、裁剪、AdamW moments 对照通过；6 步训练、保存和 6→8 恢复通过；61 passed / 1 dependency skip | `codex/speco-veomni-drafter` / `6df26e4` |
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
