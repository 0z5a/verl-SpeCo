# C5 — L20 原生 vLLM 环境

提交者：0z5a。按用户指定，C5 后续使用 L20 主机已有安装。以前容器 vLLM 0.29.0 的结果保留为历史证据，不代替此环境的验证。

| 项目 | 实际值 |
|---|---|
| Python | `/home/kxqandccx/miniconda3/envs/agent_use/bin/python`，3.13 |
| vLLM | 0.18.0，原生 site-packages |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| Transformers | 5.12.0 |
| 本轮设备 | L20，baseline 使用 GPU 4；候选与 hot/cold oracle 使用 GPU 5 |
| 原生 loader SHA-256，测试前后相同 | `b5017c1c7e4210cce28786d10fd41179d22f0d285c366df57db4caacf2355e68` |

## 兼容性与结果

原生 0.18 的 fixture 适配涉及三个已核实的接口：`pard_token=mask_token_id`；顶层 `eagle_aux_hidden_state_layer_ids=[1,2,3]`；通过 `worker.model_runner.drafter.model` 读取 drafter。0.29 fixture 的嵌套辅助层配置在 0.18 被忽略，四层 tiny target 的默认选层含重复项，曾造成 128 与 192 的特征宽度不匹配。相关失败日志保留，没有算作 loader 测试通过。

候选仍只有一行 `model.` 前缀幂等修改。`prepare_native_loader_overlay.py` 在任务目录建立独立源码副本，其余 vLLM 文件链接到原生安装，依赖和二进制扩展继续使用原生环境。共享安装没有被修改。

| 对照 | 原生 baseline | 本地候选 | 速度提升 |
|---|---|---|---|
| 原生 public loader / 生成对照 | 复现重复前缀导致的 `KeyError: model.embed_tokens.weight`，预期失败检查通过 | 连续两次完整 public load 通过；fc.weight 精确变为 0.5 倍；更新前后 2 × 16 tokens 相同 | N/A，baseline 功能失败，不能比较等量成功工作 |
| 热更新 B 对独立冷启动 B，首个 draft forward 的 5 × 256 logits | 冷启动 B reference | 仅前缀补丁失败：max abs error = 0.0556640625；target hash 与最终 tokens 相同 | N/A，正确性门槛未通过 |
| 测试侧原地刷新 proposer mask cache 后的 hot-B / cold-B | 同一独立冷启动 B reference | 1,280 logits 完全一致，max abs error = 0；target hash / tokens 相同 | N/A，诊断正确性验证 |
| EAGLE3 / DFlash 完整在线 RL | 未完成 | 未完成 | N/A |

此前超时的导入阶段采样为 `D (disk sleep)` / `folio_wait_bit_common`；本轮定位到 transformers 扫描模型源码，随后启动恢复。启动超时放宽至 900 s，不计作性能收益。测试改用标准库异常断言，并按原生 0.18 的 `KeyError` 类型及具体参数名检查失败。GPU 4 在 baseline 结束后被其他任务占用，候选首次启动因剩余显存不足而失败；换用 GPU 5 后通过。所有失败日志保留。

本机模型 registry 包含 `Eagle3LlamaForCausalLM`，但没有 `DFlashLagunaForCausalLM`、`DFlashQwen3ForCausalLM`，也未发现 DFlash 模型实现文件。原始 C5 的双算法完整验证在此安装上仍有能力缺口，没有静默更换或升级环境。

## 热更新派生缓存调查

`check_native_loader_oracle.py` 在不同进程分别加载 A 后热更新 B，以及直接冷启动 B。B 只将 fc.weight 乘 0.5；两侧先完成一次生成，再捕获同一提示的第一个 draft forward。冷启动比较覆盖 5 × 256 个 BF16 logits，前三行完全相同，最后两行最大差异分别约为 0.055664 和 0.050293。两侧 target 参数 SHA-256 相同，且热更新前后保持不变；最终 16 个 target-verified tokens 也相同。因此最终文本相同不足以证明 draft 热更新正确。

原生 `vllm/v1/spec_decode/eagle.py` 的模型加载阶段，用 `combine_hidden_states(mask_hidden)` 初始化 `parallel_drafting_hidden_state_tensor`。public model loader 更新 fc 后不会重算 proposer 中的这份缓存。测试另设 `hot-refreshed`：在测试侧原地重算缓存，验证 buffer 地址不变，并与独立 cold-B logits 比较。刷新前缓存最大误差为 0.291015625；刷新后全部 1,280 logits 精确一致，buffer 地址不变。此诊断不修改共享 vLLM，也不是生产修复；它揭示仅前缀补丁不足以支持该 parallel-drafting fixture 的热更新。

## 复现与证据

`experiment/l20/run_native_public_loader.sh` 接收任务目录及原生 Python 路径；依次运行未修改安装与独立候选副本。`check_public_loader_update.py` 记录 worker 的 Python、loader 文件路径与源码 hash，并检查两次 public load、实际 fc 权重变化和更新前后生成。

先在该原生环境中，用 `prepare_tiny_peagle.py` 重新生成 `models/`；然后用 `prepare_native_loader_overlay.py` 创建尚不存在的 `overlay/`，再运行：

```bash
bash run_native_public_loader.sh \
  /home/kxqandccx/0z5a/speco-l20-20260919/c5-native \
  /home/kxqandccx/miniconda3/envs/agent_use/bin/python
```

首次生成 fixture 时，PYTHONPATH 指向本任务已有 SpeCo 源码。overlay 已存在时直接复用，原生 loader 的 SHA 必须与记录一致。本轮属于 TP1 / BF16 / eager 的 tiny P-EAGLE checkpoint loader 正确性测试；没有计时收益结论，也没有把它标作原 PR10 IPC 或真实 EAGLE3/DFlash 训练闭环完成。

冷启动专项使用同一原生 Python 和 overlay，依次执行 `check_native_loader_oracle.py hot ROOT`、`cold ROOT`、`compare ROOT`；仅前缀候选的 compare 预期失败并保存误差报告。诊断缓存刷新使用 `hot-refreshed ROOT`，再 `compare ROOT --hot-mode hot-refreshed`。设置 `CUDA_VISIBLE_DEVICES=5`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_ALLOW_INSECURE_SERIALIZATION=1`、`PYTHONPATH=ROOT/overlay`。

原始日志、环境、registry 记录和候选 hash 位于 `evidence/l20-20260920/native/`。Shell 语法、Ruff 和 overlay helper 的 mypy 检查通过。本轮 5 个 tiny 权重文件已清理，共 2,786,856 bytes；清单见 `evidence/l20-20260920/native/resume-0910/evidence/resume-cleanup.json`。配置、logits、原始日志与 SHA-256 保留。共享原生 loader 的 SHA-256 再次核对未变。

本轮 `git fetch origin main` 成功，上游仍为 `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3`，已有工作基于该提交。原生环境中还未安装 verl / Ray，因此完整训练闭环不能由本轮 public-loader 专项替代。DFlash 后续需要支持其模型架构的原生环境；当前等待用户确认环境范围。
