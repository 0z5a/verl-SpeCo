#!/usr/bin/env bash
set -euo pipefail
cd /experiment
export UV_CACHE_DIR=/experiment/.cache/uv
export HF_HOME=/experiment/.cache/huggingface
if [ ! -x .venv-clean/bin/python ]; then
    uv venv --system-site-packages --python /usr/bin/python3 .venv-clean
fi
uv pip install --python .venv-clean/bin/python --no-deps verl==0.9.0
uv pip install --python .venv-clean/bin/python --no-deps hydra-core==1.3.2 omegaconf==2.3.0 pytest==8.3.5 ray==2.54.0 tensordict==0.10.0 datasets==4.5.0 codetiming==1.4.0 dill==0.4.0 pandas==2.3.3 pyarrow==23.0.1 wandb==0.24.2 peft==0.18.1 torchdata==0.11.0 pylatexenc==2.10 math-verify==0.8.0 antlr4-python3-runtime==4.9.3 cloudpickle==3.1.2 huggingface-hub==1.7.2 orjson==3.12.0 pybase64==1.5.0
uv pip install --python .venv-clean/bin/python --no-deps -e .
uv pip install --python .venv-clean/bin/python --target online-deps --no-deps transformers==5.10.4 tokenizers==0.22.2
.venv-clean/bin/python -m pytest -q tests/unit tests/integration/test_peagle_backend_contract.py tests/integration/test_drafter_runtime_control_contract.py tests/integration/test_native_draft_update_contract.py --junitxml=evidence/l20-20260919/baseline.xml
