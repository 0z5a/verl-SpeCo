# Standalone VeOmni dense P-EAGLE drafter

Author: 0z5a

Rebased on `4029c80b501e6bdd275b928e28bc54cf8a8537cc` on 2026-09-22.

| Current check | Result |
|---|---|
| `pytest tests/integration/test_veomni_drafter_selection.py tests/integration/test_peagle_backend_contract.py tests/unit/test_fsdp_shard_size.py -q` | 24 + 15 passed |
| Ruff on changed Python files | Passed |
| `git diff --check` | Passed |

| Historical standalone six-step lifecycle | Duration |
|---|---:|
| Flat FSDP2 | 49.401 s |
| VeOmni | 26.097 s |

The observed duration is 47.17% lower, but these are uncontrolled sequential runs on a shared host and do not establish a speedup. Historical source: 8298042. Checkpoint/resume and TP1/TP2 serving were previously exercised. The rebase preserves main's new DDP branch when the VeOmni engine is disabled. Standalone dense P-EAGLE on a full-world FSDP2 mesh only; GPU E2E has not been rerun after rebasing.

Raw logs and dependency overlays remain local and are excluded from this PR and its commits.
