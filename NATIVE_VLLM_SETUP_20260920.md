# L20 native vLLM 0.29.0

Submitter: 0z5a. The latest stable PyPI release was checked on 2026-09-20;
version 0.29.0 was downloaded and installed in a separate native environment.

| Item | Verified value |
|---|---|
| Native Python | `/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest/.venv/bin/python` |
| Python / vLLM | 3.13.14 / 0.29.0 |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
| Transformers | 5.17.0 |
| Dependency check | 196 installed packages compatible (`uv pip check`) |
| vLLM wheel | `vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl`, 315,961,042 bytes |
| Wheel SHA-256, matches official PyPI metadata | `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635` |
| DFlash entries in wheel registry | `DFlashLagunaForCausalLM` and `DFlashQwen3ForCausalLM` present |

The original `agent_use` environment remains separate. Downloads used the
Tsinghua PyPI mirror after the initial official-index attempt stalled. The
wheel was independently checked against the official PyPI SHA-256. Its retained
copy is under `c5-native-latest/wheels/`. This is a package artifact, not model
weights. Runtime imports report the new native environment; no container
interpreter is used for C1 serving.

Initial imports encountered filesystem write waits. The successful runtime
check and C1 driver use `PYTHONDONTWRITEBYTECODE=1`. Raw installation and runtime
records are in `evidence/l20-20260920/native-latest/`. Registry presence does not
by itself establish a DFlash E2E pass. The C1 matrix uses a six-step FSDP2-trained
P-EAGLE checkpoint from the C2 fix, with independent frozen serving validation.

## C3 coordination

The requested coordination comment was posted by `0z5a` on Roadmap #7:
https://github.com/verl-project/verl-SpeCo/issues/7#issuecomment-5746767473

It requests the existing bubble-time branch/PR and the interface boundary with
#85, and offers review of fixed worker groups, budgets, optimizer boundaries,
weight versions and E2E timing. It reports the current VeOmni standalone checks
and explicitly leaves online train–publish–rollout validation pending.
