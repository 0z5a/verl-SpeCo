# PR10 native L20 E2E status

Author: 0z5a. Original PR production source:
`b258ec517977a01df722909789da42746c39f28f`. Only experiment harness files are
added in this checkout; the PR loader implementation is unchanged.

**Not completed; SSH access is currently unavailable.** DFlash v9 was last
confirmed at 9/20 steps, writing the step-10 checkpoint. EAGLE3 v17 was queued
after it. The run audits public-loader completion and whether private
`fc.weight` survives later target synchronization.

| Check | Baseline | Candidate | Speed change | Result |
|---|---|---|---|---|
| Native EAGLE3 20-step online RL | One effective update before transport failure | v17 queued, current state unavailable | N/A | v11 exited before optimizer update: missing padding dependency; v15 uses upstream padding helpers, a longer-response fixture and unpadded single-sequence microbatches |
| Native DFlash 20-step online RL | Not measured | Last confirmed 9/20 steps | N/A | Repeated public loads pass; intermediate FC retention fails; final result unavailable |
| Actual PR10 public-loader audit | Original source | Logging and FC retention hash | N/A | Test overlay installed; no loader logic changes |

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
Inspect `evidence/full-e2e/online-eagle-pr10-v17.{log,exit}` and
`online-dflash-pr10-v9.{log,exit}` before restarting. The same detached launcher
runs them sequentially with 60-minute bounds on GPUs 0/1. Final completion,
nonzero optimizer updates, publication/retention counts and model cleanup remain
pending.

The full target state is 8,822,848,512 bytes including tied aliases. vLLM 0.29
rejects an isolated `lm_head.weight` bucket when its tied embedding is absent
from that invocation. The candidate uses a 9,216 MiB actor transfer bucket and a
separate 128 MiB draft bucket. v8's failure and stopped process tree are retained;
v9 was interrupted after checking the exact byte requirement, and v10 was moved
because another workload reduced GPU 4's free memory.

`prepare_loader_audit.py` copies the original package into an isolated test overlay,
records runtime source hashes, logs successful public loads, and compares a
private FC weight hash after later target synchronization. The FC hash is a
sampled retention check, not a complete parameter or derived-cache oracle.

Startup mitigation is isolated to this task's PYTHONPATH: a tmpfs copy of all
2,434 Transformers files has identical hashes; the 235-package distribution map
was captured once from the pinned environment and cached for new processes.
`metadata_cache_sitecustomize.py` uses that map and normal Ray worker priority.
Existing task workers and their threads were restored from nice 15 to normal 0;
other jobs were not changed. Do not reuse the metadata cache after changing
installed packages. These instrumentation/startup changes have no controlled
speed comparison. Ruff check/format and shell syntax checks pass locally.

EAGLE3 v11 and DFlash v4 both completed initial target synchronization and reached
old-logprob computation after rollout, then exited 1 because verl imports
`flash_attn.bert_padding` even with actor SDPA. The isolated padding overlay
redirects that import to the unchanged upstream pure-PyTorch module at
`Dao-AILab/flash-attention@edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d`.
Unpadding values, padding values and backward gradients match the direct masked
reference. This does not install or claim validation of FlashAttention CUDA kernels.

v12 was stopped during model initialization: Ray prepends the driver's working
directory, which could shadow the audit package with the original checkout.
v13 launches from the audit directory and logs each imported runtime path.
Earlier runs establish only their observed execution/failure, not successful
audited publication. Raw failure logs and the v12 stopped-process manifest are retained.

v13 confirmed the audit source on Ray workers and passed the missing-padding
boundary. Its first completed step had five-token answers, zero collected draft
samples, zero draft updates and zero actor gradient. The run was stopped; this
is not an E2E success. v14/v7 use explained multiplication prompts and a two-row
minimum capture window to exercise training on shorter responses. The existing
length reward term is unchanged. The result checker requires nonzero actor
gradient, successful draft updates at all 20 steps, all-rank public loads,
completed RPCs and subsequent FC retention, in addition to process completion.

v14 reached real hidden-state collection and exited with the original PR's
requirement that `use_remove_padding=True` and `DatasetPadMode.NO_PADDING`.
v15/v8 enable it with SDPA, fixed per-GPU microbatch size 1 and dynamic batching
disabled for both actor updates and old-logprob inference. A patched tiny Qwen3
reference check compares padded and unpadded single-sequence forward outputs:
maximum FP32 error 8.9406967e-8. This does not validate SDPA packing multiple
independent documents in one sequence; the harness explicitly avoids that case.

v15 completed one effective training step: actor gradient norm 19.875,
2 collected drafter samples and 1 successful draft optimizer step. Its subsequent
CUDA IPC publication failed on both ranks with `pidfd_getfd: Operation not permitted`.
The async `drafter/published=1` metric therefore did **not** mean completion.
The failed RPC and owned stopped-process manifest are retained. v16/v9 select the
original PR's `draft_update_use_shm=True` transport while preserving the same
native public loader. Host security policy is unchanged. The per-algorithm bound
is now 60 minutes to accommodate actual shared-node step time and checkpoint I/O.

v16 completed one effective EAGLE3 update but the shared-memory sender rejected
its 163,840,000-byte BF16 `lm_head.weight`: 128 MiB cannot hold this tensor.
The next EAGLE3 run uses 256 MiB. DFlash v9 continues with 128 MiB because its
largest published checkpoint tensor is the 65,536,000-byte FC weight. The native
loader has completed 58-tensor publications on both ranks, and subsequent steps
have trained and published changed FC fingerprints.

DFlash's target-sync retention check fails: the observed post-sync FC hash equals
the initial safetensors FC hash (`7ea10cdf…3ca673`), not the most recently published
hash. The unconditional checkpoint reload is already present in PR10's parent;
this is not attributed to the public-loader change. With publish interval 1 a
new draft is published after each target sync, so the 20-step run continues.
The report separates training/publication completion from intermediate retention.

Original PR10 waits for pending publication RPCs before each rollout and in
`fit` cleanup. Final completion evidence therefore requires all-rank load counts,
all 20 steps and a clean exit through those barriers. Adapter completion logs are
recorded when present; async `published=1` alone never satisfies this check.

Connection status at 2026-09-20 14:16–14:18 Asia/Shanghai: SSH port 6000
refuses new connections. Last confirmed DFlash v9 state was 9 completed steps,
with step 10 writing checkpoints (about 5.6 GiB present); both ranks had executed
the native public loader repeatedly. EAGLE3 v17 was queued behind PID 4101336
by PID 4173853. Remote completion and cleanup remain unverified until access
returns; these are detached processes, not a reason to start duplicate jobs.
