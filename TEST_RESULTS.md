# PR10 native online E2E results

Author: 0z5a

Both native vLLM runs completed 20 online RL steps from scratch on two L40S GPUs. Each step performed an actor update, drafter training and asynchronous drafter publication. Both processes exited with status 0.

| Algorithm | Steps | First step | Steady mean (steps 2–20) | Sum of step times | Relative steady speed |
|---|---:|---:|---:|---:|---:|
| DFlash | 20/20 | 42.134 s | 22.399 s | 467.706 s | 1.000× |
| EAGLE3 | 20/20 | 33.394 s | 14.397 s | 306.934 s | 1.556× |

EAGLE3's observed steady step time was 35.72% lower than DFlash's in this fixture. This compares different drafter architectures under the same harness; it does not isolate a code optimization or establish a production throughput claim. The sum covers logged training steps and excludes process startup and shutdown.

| Validation | DFlash | EAGLE3 |
|---|---:|---:|
| Exit status | 0 | 0 |
| Global-step range | 1–20 | 1–20 |
| Successful drafter train metrics | 20 | 20 |
| Successful publish metrics | 20 | 20 |
| Post-step retention checks | 0/38 retained | 0/38 retained |

The retention audit confirms a known lifecycle defect: a later target sync restores the checkpoint drafter instead of retaining the most recently trained drafter. The behavior also exists in PR10's parent. Publication interval 1 loads a newly trained drafter again after every target sync, allowing both 20-step runs to finish, but it does not fix the retention defect.

The test used Python 3.12.14, PyTorch 2.13.0+cu129, vLLM 0.29.0, Transformers 5.17.0 and verl commit `7aed6b230776f963fa09509c10d9c3a767d1102c`. The serving environment was native to the machine. The tested source commit was `cf297c4a1ce806c2bc6114483fc63b86ef24e589`.

Raw logs, exit files, timing JSON, environment versions and pre-cleanup model hashes are under `evidence/l40s-20260920/`. Both successful runs emitted a DataLoader worker termination traceback during shutdown after reaching 20/20; the wrappers still completed their publication barriers and exited 0.
