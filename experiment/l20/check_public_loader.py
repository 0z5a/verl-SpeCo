"""CUDA reproduction of PR #10's required EAGLE3 loader dependency."""

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
from vllm import LLM, SamplingParams


def probe(worker):
    draft = worker.get_draft_model()
    weight = draft.model.fc.weight.detach().clone()
    draft.load_weights([("fc.weight", weight)])
    with pytest.raises(
        ValueError, match="no module or parameter named 'model'"
    ) as failure:
        draft.load_weights([("model.fc.weight", weight)])
    torch.testing.assert_close(draft.model.fc.weight, weight, rtol=0, atol=0)
    source = inspect.getsource(type(draft).load_weights)
    return {
        "model_class": type(draft).__name__,
        "loader_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "unprefixed_load": "PASS",
        "prefixed_load": "FAIL",
        "error": str(failure.value),
        "parameter_unchanged": True,
    }


if __name__ == "__main__":
    llm = LLM(
        model="/experiment/models/target",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.6,
        speculative_config={
            "model": "/experiment/models/eagle",
            "method": "eagle3",
            "num_speculative_tokens": 3,
        },
    )
    outputs = llm.generate(
        ["What is 2 + 2?"], SamplingParams(temperature=0, max_tokens=16)
    )
    report = {
        "generated_token_count": len(outputs[0].outputs[0].token_ids),
        "loader": llm.collective_rpc(probe),
    }
    Path("/experiment/evidence/l20-20260919/public-loader.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
