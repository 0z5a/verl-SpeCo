# PR10 native L20 E2E status

Author: 0z5a. Original PR production source:
`b258ec517977a01df722909789da42746c39f28f`. Only experiment harness files are
added in this checkout; the PR loader implementation is unchanged.

**Not completed.** The L20 SSH endpoint stopped accepting connections during
the run. EAGLE3 v8 was last observed importing Transformers; DFlash v1 was
queued after its exit. Neither is recorded as passing.

| Check | Baseline | Candidate | Speed change | Result |
|---|---|---|---|---|
| Native EAGLE3 20-step online RL | Not completed | Not completed | N/A | v7 initialization OOM; v8 final status unavailable |
| Native DFlash 20-step online RL | Not completed | Not completed | N/A | Queued; final status unavailable |
| Actual PR10 public-loader audit | Uninstrumented | Audit overlay prepared | N/A | Not executed |

The separate native environment uses vLLM 0.29.0, PyTorch 2.13.0+cu130,
Transformers 5.10.4, Python 3.13.14 and a verl 0.8.0 import overlay. The vLLM
test dependency overlay adds the previously tested EAGLE3 idempotent model-name
prefix guard and `FusedMoE = RoutedExperts` solely for dense-model import
compatibility. This is a local dependency candidate, not an author-supplied
companion commit and not MoE/FP8 validation. Shared installed packages are not
patched by the overlay.

The old training config loader requires `LlamaForCausalLMEagle3`; a separate
config directory provides that name and symlinks unchanged weights. The vLLM
speculative-config override continues to point to the original serving model.
Synthetic arithmetic reward includes a small length term to exercise GRPO;
it is not a model-quality benchmark.

Remote experiment root:
`/home/kxqandccx/0z5a/speco-l20-20260919/c5-native-latest`.
Inspect `evidence/full-e2e/online-eagle-pr10-v8.{log,exit}` and
`online-dflash-pr10-v1.{log,exit}` before restarting. The latter is launched by
the same detached shell after EAGLE3 exits. Revalidate running processes and GPU
availability. These jobs still need raw evidence retrieval, actual-loader-path
verification, effective actor/drafter updates, later-rollout validation, and
completed-model cleanup.

`prepare_loader_audit.py` creates a test-only source overlay with one log after
the original public loader returns and records both source hashes. It has not
been uploaded or executed. `SPECO_AUDIT_OVERLAY` and `SPECO_PYTHON_CACHE` are
optional launcher paths; leave them unset unless the corresponding directories
have been fully prepared. Ruff check/format and shell syntax checks pass locally.
