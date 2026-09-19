"""Inspect EAGLE3 head ownership after the vLLM proposer initializes."""

import json
from pathlib import Path

from vllm import LLM


def inspect_head(worker):
    draft = worker.get_draft_model()
    target = worker.model_runner.model
    return {
        "rank": worker.rank,
        "runner_type": type(worker.model_runner).__module__,
        "draft_type": type(draft).__name__,
        "draft_vocab_size": draft.config.draft_vocab_size,
        "head_shape": list(draft.lm_head.weight.shape),
        "shares_target_head": draft.lm_head is target.lm_head,
        "head_in_named_parameters": "lm_head.weight" in dict(draft.named_parameters()),
        "head_aliases": [
            name
            for name, param in draft.named_parameters(remove_duplicate=False)
            if param is draft.lm_head.weight
        ],
    }


if __name__ == "__main__":
    llm = LLM(
        model="/experiment/models/target",
        tensor_parallel_size=2,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.25,
        speculative_config={
            "model": "/experiment/models/eagle",
            "method": "eagle3",
            "num_speculative_tokens": 3,
        },
    )
    report = llm.collective_rpc(inspect_head)
    Path("/experiment/evidence/l20-20260919/eagle-head-ownership.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
