# Checkpointed P-EAGLE sequence partitions

Author: 0z5a

Rebased on `4029c80b501e6bdd275b928e28bc54cf8a8537cc` on 2026-09-22.

| Current check | Result |
|---|---|
| `pytest tests/integration/test_peagle_partition.py -q` | 27 passed |
| Ruff on changed Python files | Passed |
| `git diff --check` | Passed |

| Historical Qwen3-4B optimizer comparison | Baseline | Candidate | Observed change |
|---|---:|---:|---:|
| Old partition → pruned projection partition | 2021.38 ms | 1692.40 ms | 19.44% higher step rate |
| Flat → pruned projection partition | 1317.97 ms | 1692.40 ms | 22.12% lower step rate |
| Peak allocated memory, old → fixed partition | 16.28 GiB | 14.23 GiB | 12.59% lower using rounded values |

Historical measurements are shared-node observations from ae8b5a2, not fresh performance validation of this revision. The independent PR excludes the former VeOmni integration dependency. Loss, gradient and optimizer parity tests cover packed documents, reduced/full vocabulary, partition counts and empty loss masks. Current local CPU tests use PyTorch 2.8.0; an earlier different environment failed FlexAttention CPU backward. GPU E2E must be repeated before merge.

Raw logs and dependency overlays remain local and are excluded from this PR and its commits.
