# Frozen P-EAGLE export

Author: 0z5a

Rebased on `4029c80b501e6bdd275b928e28bc54cf8a8537cc` on 2026-09-22.

| Current check | Result |
|---|---|
| `pytest tests/unit/test_convert_peagle_vllm.py -q` | 7 passed |
| Ruff on changed Python files | Passed |
| `git diff --check` | Passed |

| Historical validation | Result |
|---|---|
| TP1/TP2, eager/graph, target-only/draft | Generated tokens matched |
| Cached logits | 64/64 within tolerance; 8/64 strict argmax differ on BF16 ties |

No timed performance benefit is claimed. The converter copies weights unchanged and updates configuration for native EAGLE3 parallel drafting. Full vocabulary only. Historical GPU runs used source e813ce5; the rebased source has not had a new GPU E2E run.

Raw logs and dependency overlays remain local and are excluded from this PR and its commits.
