"""Compare a hot-loaded draft against an independently cold-loaded checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file
from vllm import LLM, SamplingParams

from check_public_loader_update import update


def target_hash(worker):
    digest = hashlib.sha256()
    for name, parameter in worker.model_runner.model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def refresh_parallel_mask(worker):
    proposer = worker.model_runner.drafter
    cached = proposer.parallel_drafting_hidden_state_tensor
    expected = proposer.model.combine_hidden_states(
        proposer.model.mask_hidden.view(3 * proposer.hidden_size)
    )
    error = (cached - expected).abs().max().item()
    address = cached.data_ptr()
    cached.copy_(expected)
    assert cached.data_ptr() == address
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    return {"stale_cache_max_error": error, "address_preserved": True}


def capture_first_logits(worker, path):
    draft = worker.model_runner.drafter.model

    def capture(module, inputs, kwargs, outputs):
        torch.save(module.compute_logits(outputs[0]).detach().cpu(), path)
        handle.remove()

    handle = draft.register_forward_hook(capture, with_kwargs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hot", "hot-refreshed", "cold", "compare"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--hot-mode", choices=["hot", "hot-refreshed"], default="hot")
    args = parser.parse_args()
    evidence = args.root / "evidence"
    if args.mode == "compare":
        hot = torch.load(evidence / f"{args.hot_mode}-logits.pt", weights_only=True)
        cold = torch.load(evidence / "cold-logits.pt", weights_only=True)
        reports = [
            json.loads((evidence / f"{mode}-oracle.json").read_text())
            for mode in (args.hot_mode, "cold")
        ]
        assert reports[0]["tokens"] == reports[1]["tokens"]
        assert reports[0]["target_hash"] == reports[1]["target_hash"]
        report = {
            "logit_shape": list(hot.shape),
            "max_abs_error": (hot - cold).abs().max().item(),
            "tokens_equal": True,
            "target_equal": True,
        }
        (evidence / f"{args.hot_mode}-cold-oracle.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report))
        torch.testing.assert_close(hot, cold, rtol=0, atol=0)
        raise SystemExit(0)

    draft_path = args.root / "models/draft-eagle3"
    if args.mode == "cold":
        cold_path = args.root / "models/draft-cold-b"
        cold_path.mkdir(exist_ok=True)
        shutil.copyfile(draft_path / "config.json", cold_path / "config.json")
        weights = load_file(str(draft_path / "model.safetensors"))
        weights["fc.weight"] *= 0.5
        save_file(weights, str(cold_path / "model.safetensors"))
        draft_path = cold_path
    llm = LLM(
        model=str(args.root / "models/target"),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        speculative_config={
            "model": str(draft_path),
            "method": "eagle3",
            "num_speculative_tokens": 3,
            "parallel_drafting": True,
        },
    )
    prompts = [{"prompt_token_ids": [1, 4, 7]}]
    sampling = SamplingParams(temperature=0, max_tokens=16)
    llm.generate(prompts, sampling)
    before = llm.collective_rpc(target_hash)
    loader = []
    if args.mode in {"hot", "hot-refreshed"}:
        loader = llm.collective_rpc(
            update,
            kwargs={
                "expect_failure": False,
                "checkpoint": str(args.root / "models/draft-original"),
                "legacy_worker": True,
            },
        )
    refresh = (
        llm.collective_rpc(refresh_parallel_mask)
        if args.mode == "hot-refreshed"
        else []
    )
    llm.collective_rpc(
        capture_first_logits, kwargs={"path": str(evidence / f"{args.mode}-logits.pt")}
    )
    outputs = llm.generate(prompts, sampling)
    after = llm.collective_rpc(target_hash)
    assert before == after
    report = {
        "mode": args.mode,
        "target_hash": after,
        "target_unchanged": True,
        "tokens": outputs[0].outputs[0].token_ids,
        "loader": loader,
        "refresh": refresh,
    }
    (evidence / f"{args.mode}-oracle.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
