# Native L40S E2E evidence

| Algorithm | Completed steps | First step | Steady mean | Mean including first | Total logged step time | Steady comparison |
|---|---:|---:|---:|---:|---:|---:|
| DFlash | 20/20 | 42.134 s | 22.399 s | 23.385 s | 467.706 s | baseline |
| EAGLE3 | 20/20 | 33.394 s | 14.397 s | 15.347 s | 306.934 s | 35.72% lower; 1.556× faster |

The comparison uses steps 2–20 for the steady mean. It is an observed comparison between two drafter architectures in one synthetic online RL fixture, not an isolated optimization benchmark.

Both exit files contain `0`. Both logs contain global steps 1 through 20, 20 `drafter/trained:1.0` metrics and 20 `drafter/published:1.0` metrics. All 38 post-step, all-rank retention checks per algorithm report `latest_draft_retained=False`.

Files in `native/` are covered by `SHA256SUMS`. Model file hashes were recorded before task-owned model cleanup.
