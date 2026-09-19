# C5 — L20 原生 vLLM 环境

提交者：0z5a。按用户指定，C5 后续使用 L20 主机已有安装。以前容器 vLLM 0.29.0 的结果保留为历史证据，不代替此环境的验证。

| 项目 | 实际值 |
|---|---|
| Python | `/home/kxqandccx/miniconda3/envs/agent_use/bin/python`，3.13 |
| vLLM | 0.18.0，原生 site-packages |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| Transformers | 5.12.0 |
| 本轮设备 | L20，`CUDA_VISIBLE_DEVICES=5` |
| 原生 loader SHA-256，测试前后相同 | `b5017c1c7e4210cce28786d10fd41179d22f0d285c366df57db4caacf2355e68` |

## 兼容性与结果

原生 0.18 的 fixture 适配涉及三个已核实的接口：`pard_token=mask_token_id`；顶层 `eagle_aux_hidden_state_layer_ids=[1,2,3]`；通过 `worker.model_runner.drafter.model` 读取 drafter。0.29 fixture 的嵌套辅助层配置在 0.18 被忽略，四层 tiny target 的默认选层含重复项，曾造成 128 与 192 的特征宽度不匹配。相关失败日志保留，没有算作 loader 测试通过。

候选仍只有一行 `model.` 前缀幂等修改。`prepare_native_loader_overlay.py` 在任务目录建立独立源码副本，其余 vLLM 文件链接到原生安装，依赖和二进制扩展继续使用原生环境。共享安装没有被修改。

| 对照 | 原生 baseline | 本地候选 | 速度提升 |
|---|---|---|---|
| 修正 fixture 后的完整 loader / 生成对照 | Python 导入时等待文件页，300 s 超时，exit 124 | 未执行：baseline 未完成 | N/A，不能申报收益 |
| EAGLE3 / DFlash 完整在线 RL | 未完成 | 未完成 | N/A |

超时进程采样为 `D (disk sleep)`，wchan 为 `folio_wait_bit_common`，没有进入该次 GPU 计算。超时后进程已退出。更早一轮已进入生成，其 worker accessor 不兼容随后被修正；这也不构成最终通过。

本机模型 registry 包含 `Eagle3LlamaForCausalLM`，但没有 `DFlashLagunaForCausalLM`、`DFlashQwen3ForCausalLM`，也未发现 DFlash 模型实现文件。原始 C5 的双算法完整验证在此安装上仍有能力缺口，没有静默更换或升级环境。

## 复现与证据

`experiment/l20/run_native_public_loader.sh` 接收任务目录及原生 Python 路径；依次运行未修改安装与独立候选副本。`check_public_loader_update.py` 记录 worker 的 Python、loader 文件路径与源码 hash，并检查两次 public load、实际 fc 权重变化和更新前后生成。

先在该原生环境中，用 `prepare_tiny_peagle.py` 重新生成 `models/`；然后用 `prepare_native_loader_overlay.py` 创建尚不存在的 `overlay/`，再运行：

```bash
bash run_native_public_loader.sh \
  /home/kxqandccx/0z5a/speco-l20-20260919/c5-native \
  /home/kxqandccx/miniconda3/envs/agent_use/bin/python
```

首次生成 fixture 时，PYTHONPATH 指向本任务已有 SpeCo 源码。overlay 已存在时直接复用，原生 loader 的 SHA 必须与记录一致。正确性未完成前不进行性能采样。

原始日志、环境、registry 记录和候选 hash 位于 `evidence/l20-20260920/native/`。Shell 语法、Ruff 和 overlay helper 的 mypy 检查通过。本轮 4 个 tiny 权重文件已清理，共 2,271,968 bytes；配置和 SHA-256 清单保留。
