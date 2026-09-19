"""Fresh-process full drafter optimizer-step timing on real target features."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--partitions", type=int, default=1)
parser.add_argument("--tokens", type=int, default=256)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, args.source)
from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend  # noqa: E402

torch.manual_seed(11)
tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
target = (
    AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    .to("cuda")
    .eval()
)
target_cfg = target.config
aux_layers = [2, target_cfg.num_hidden_layers // 2, target_cfg.num_hidden_layers - 3]
texts = [
    "Explain how a computer executes instructions. ",
    "Describe the water cycle and why rain falls. ",
]
ids, aux, last = [], [], []
for text in texts:
    tokens = tokenizer((text * args.tokens), return_tensors="pt")["input_ids"][
        :, : args.tokens // 2 + 1
    ].to("cuda")
    with torch.no_grad():
        output = target(tokens, output_hidden_states=True)
    ids.append(tokens[0, 1:])
    aux.append(torch.cat([output.hidden_states[i][0, :-1] for i in aux_layers], dim=-1))
    last.append(output.hidden_states[-1][0, 1:])
batch = {
    "input_ids": torch.cat(ids).unsqueeze(0),
    "hidden_states": torch.cat(aux).unsqueeze(0),
    "last_hidden_states": torch.cat(last).unsqueeze(0),
    "seq_lengths": torch.tensor([x.numel() for x in ids], device="cuda"),
}
batch["loss_mask"] = torch.ones_like(batch["input_ids"])
batch["attention_mask"] = torch.ones_like(batch["input_ids"])
input_hash = hashlib.sha256(batch["input_ids"].cpu().numpy().tobytes()).hexdigest()
del target, output, ids, aux, last
torch.cuda.empty_cache()
config = OmegaConf.create(
    {
        "model": {"path": args.target},
        "rollout": {
            "drafter": {
                "speculative_algorithm": "PEAGLE",
                "model_path": "",
                "training": {
                    "use_logits": False,
                    "peagle_num_draft_layers": 2,
                    "peagle_num_aux_hidden_states": 3,
                    "peagle_num_depths": 8,
                    "peagle_sequence_partitions": args.partitions,
                    "lr": 1e-4,
                },
            }
        },
    }
)
torch.manual_seed(123)
backend = PEagleTrainerBackend(config, target_cfg)
model, _ = backend.build_model()
model = model.to(device="cuda", dtype=torch.bfloat16).train()
backend.target_model = backend.target_model.to(device="cuda", dtype=torch.bfloat16)
optimizer = backend.setup_optimizer(model, config.rollout.drafter.training)
timings, losses, counts = [], [], []
for step in range(8):
    torch.manual_seed(900 + step)
    torch.cuda.synchronize()
    if step == 3:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    result = backend.compute_loss(model, batch, 0)
    loss = result["total_local_ploss"] / result["local_num_tokens"].clamp_min(1)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    losses.append(float(loss.detach()))
    counts.append(float(result["local_num_tokens"]))
    if step >= 3:
        timings.append(elapsed)
    print(
        json.dumps(
            {"step": step, "seconds": elapsed, "loss": losses[-1], "tokens": counts[-1]}
        ),
        flush=True,
    )
record = {
    "scope": "target feature collection plus full drafter training; no rollout publication",
    "source": str(
        Path(sys.modules[PEagleTrainerBackend.__module__].__file__).resolve()
    ),
    "source_sha256": hashlib.sha256(
        Path(sys.modules[PEagleTrainerBackend.__module__].__file__).read_bytes()
    ).hexdigest(),
    "input_sha256": input_hash,
    "partitions": args.partitions,
    "tokens": args.tokens,
    "step_seconds": timings,
    "losses": losses,
    "sampled_counts": counts,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
}
Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
