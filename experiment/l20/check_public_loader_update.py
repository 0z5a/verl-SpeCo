"""Full-checkpoint GPU probe of the explicitly local vLLM prefix dependency patch."""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest import TestCase

import torch
from safetensors.torch import load_file
from vllm import LLM, SamplingParams


def update(worker, expect_failure, checkpoint, legacy_worker):
    draft = (
        worker.model_runner.drafter.model if legacy_worker else worker.get_draft_model()
    )
    weights = load_file(str(Path(checkpoint) / "model.safetensors"))
    weights["fc.weight"] = weights["fc.weight"] * 0.5
    names = {
        name
        if name in {"lm_head.weight", "d2t", "t2d", "mask_hidden"}
        else "model." + name: value
        for name, value in weights.items()
    }
    before = draft.model.fc.weight.detach().clone()
    if expect_failure:
        error = KeyError if legacy_worker else ValueError
        message = (
            r"model\.embed_tokens\.weight"
            if legacy_worker
            else "no module or parameter named 'model'"
        )
        with TestCase().assertRaisesRegex(error, message):
            draft.load_weights(names.items())
        torch.testing.assert_close(draft.model.fc.weight, before, rtol=0, atol=0)
    else:
        draft.load_weights(names.items())
        torch.testing.assert_close(draft.model.fc.weight, before * 0.5, rtol=0, atol=0)
        # Repeat with the same public names: a second update must remain idempotent.
        draft.load_weights(names.items())
        torch.testing.assert_close(draft.model.fc.weight, before * 0.5, rtol=0, atol=0)
    return {
        "expected_prefix_failure": expect_failure,
        "python": sys.executable,
        "loader_file": inspect.getfile(type(draft)),
        "accessor": "model_runner.drafter.model"
        if legacy_worker
        else "get_draft_model",
        "fc_weight_changed": not torch.equal(before, draft.model.fc.weight),
        "loader_sha256": hashlib.sha256(
            inspect.getsource(type(draft).load_weights).encode()
        ).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-failure", action="store_true")
    parser.add_argument("--legacy-worker-accessor", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--models-root", type=Path, default=Path("/experiment/tiny-peagle")
    )
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("/experiment/evidence/l20-20260919")
    )
    args = parser.parse_args()
    llm = LLM(
        model=str(args.models_root / "target"),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        speculative_config={
            "model": str(args.models_root / "draft-eagle3"),
            "method": "eagle3",
            "num_speculative_tokens": 3,
            "parallel_drafting": True,
        },
    )
    prompts = [{"prompt_token_ids": [1, 4, 7]}, {"prompt_token_ids": [1, 8, 9, 10, 11]}]
    sampling = SamplingParams(temperature=0, max_tokens=16)
    before = llm.generate(prompts, sampling)
    result = llm.collective_rpc(
        update,
        kwargs={
            "expect_failure": args.expect_failure,
            "checkpoint": str(args.models_root / "draft-original"),
            "legacy_worker": args.legacy_worker_accessor,
        },
    )
    after = llm.generate(prompts, sampling)
    tokens = [row.outputs[0].token_ids for row in after]
    assert tokens == [row.outputs[0].token_ids for row in before]
    report = {"loader": result, "token_ids": tokens}
    (args.evidence_root / f"{args.label}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))
