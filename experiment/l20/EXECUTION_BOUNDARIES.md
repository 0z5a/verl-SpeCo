# Remaining execution boundaries

## C3: idle-worker scheduling

The baseline `SyncExecutionStrategy.execute` submits a plan and immediately
resolves it. The worker preflight includes worker identity, data version and
target version; asynchronous publication does not make training asynchronous.

Open PR #85 (`f27a3a35001603bc32c01e5dea560d570b092e4a`) already changes the
scheduler, worker, feature store, actor-loop integration and publication for
asynchronous DSpark. Its changed-file list was checked on 2026-09-19. This run
does not introduce another asynchronous scheduler. No idle-worker speedup is
measured or claimed.

## C4: VeOmni execution

The installed environment has no `veomni` package. The inspected upstream is
ByteDance-Seed/VeOmni at `77cf73e69756eaff19c3f32df6f7415b0240e791`.

| Boundary | Observed API | Consequence |
|---|---|---|
| Parallel state | Current VeOmni exports `init_parallel_state_from_config` and `use_parallel_state`; its old initializer is now private `_init_parallel_state`. Installed verl's VeOmni engine calls `init_parallel_state`. | Current upstream cannot be assumed compatible with installed verl. |
| Wrapping | `build_parallelize_model(model: nn.Module, ...)` accepts a custom module and `_no_split_modules`. | A custom drafter wrapper is not excluded by the signature. Actual forward/backward parity remains untested. |
| Materialization | FSDP2 requires `init_device=meta`, then materializes/loads weights. | Calling this on the existing initialized wrapper is not a validated adapter; preserve wrapper parameter names, frozen target head and initialization explicitly. |
| Optimizer / checkpoint | Installed verl builds VeOmni optimizers and uses `FSDPCheckpointManager`. | Ordinary actor checkpoint support does not prove custom drafter optimizer/RNG/publish restoration. |
| Ownership | Parallel state is ambient but current upstream provides a scoped context. | Any adapter must bind the drafter's own group and restore the prior state. |

No engine configuration or production adapter was added. A version-compatible
dependency and a meta-materialization/state-loading oracle are required before
an engine injection can be tested. This is an API investigation, not VeOmni E2E.

Follow-up: verl v0.9.0 (`483b8a009ba3a97563edee3a19887e4862b8094a`)
pins **VeOmni 0.1.11** in its CUDA E2E workflow. Its wheel SHA-256 is
`fe399ead11350fa3d53cc53293cd60d0c1042f2fa15d599aeb89db9aa417b129`.
The wheel was downloaded and verified in the task's `0z5a` cache, without
changing the GPU environment. This version has the required public
`init_parallel_state` API; the latest-upstream mismatch above is therefore
not a blocker for every version. It has no scoped `use_parallel_state` API,
and repeated initialization retains the existing singleton. An initial probe
must use dedicated processes for the drafter group. FSDP2 still requires meta
initialization and either checkpoint loading or `model.init_weights()`;
the existing initialized training wrapper cannot simply be handed to it.

The pinned wheel was then installed with `--no-deps --target
/experiment/veomni-deps`; the existing runtime dependencies were not upgraded.
`check_veomni_drafter.py` ran in two dedicated L20 processes using the actual
P-EAGLE training wrapper, meta parameters, a full wrapper checkpoint, FP32,
unequal rank token counts and fixed COD seeds. Both ranks reported exact
parameter/buffer restoration and loss/gradient/SGD parity against native
FSDP2 (`atol=2e-5`, `rtol=2e-4`). Target logits were supplied separately;
no target model parameters were included in the optimizer. This resolves the
small dense-wrapper materialization question, not scheduler integration,
optimizer checkpoint/RNG recovery, BF16 performance or full RL E2E.

```bash
PYTHONPATH=/experiment/veomni-deps:/experiment/online-deps:/experiment \
  OMP_NUM_THREADS=4 /experiment/.venv-clean/bin/python -m torch.distributed.run \
  --nproc-per-node=2 --master-port=29589 /experiment/check_veomni_drafter.py
```

## C5: original public-loader PR

PR #10 head is `b258ec517977a01df722909789da42746c39f28f`, base
`09c4a23770335297654ca1cb72dafb9e6e1f8ef1`. Its original runtime source calls the
outer draft model's public loader after prefixing EAGLE3 backbone names.
Installed vLLM 0.29.0 prefixes those names again. The CUDA probe rejects
`model.fc.weight`; the PR body names a paired dependency but provides no SHA.
The source does not define current main's `weight_update_mode` option.

Current-main EAGLE3/DFlash lifecycle results therefore do not validate original
PR #10, public-loader hot/cold logits, target isolation or CUDA Graph replay.
