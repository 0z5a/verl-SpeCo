"""Two-rank FSDP2 gradient and optimizer parity with unequal token counts."""

import os
import sys
import hashlib
from pathlib import Path
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.environ["SPECO_SOURCE"])
from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel  # noqa: E402
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig  # noqa: E402

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
source = Path(sys.modules[PEagleTrainingModel.__module__].__file__).resolve()
assert source.is_relative_to(Path(os.environ["SPECO_SOURCE"]))
print(
    f"rank={rank} source={source} sha256={hashlib.sha256(source.read_bytes()).hexdigest()}",
    flush=True,
)
torch.manual_seed(17)
config = PeagleConfig(
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=2,
    num_hidden_layers=2,
    num_draft_layers=2,
    target_hidden_size=32,
    num_aux_hidden_states=3,
    vocab_size=64,
    num_depths=3,
    mask_token_id=63,
    max_position_embeddings=64,
)
flat = PEagleTrainingModel(LlamaForCausalLMPeagle(config), num_depths=3).cuda()
partitioned = deepcopy(flat)
partitioned.sequence_partitions = 3
fully_shard(flat)
fully_shard(partitioned)
torch.manual_seed(23 + rank)
mask = torch.ones(1, 12, device="cuda")
mask[:, : rank + 1] = 0
batch = dict(
    input_ids=torch.randint(0, 63, (1, 12), device="cuda"),
    aux_hidden=torch.randn(1, 12, 96, device="cuda"),
    loss_mask=mask,
    attention_mask=torch.ones_like(mask),
    target_logits=torch.randn(1, 12, 64, device="cuda"),
    seq_lengths=torch.tensor([5, 7], device="cuda"),
)
for model in (flat, partitioned):
    torch.manual_seed(41 + rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num, den, _ = model(**batch)
    global_den = den.detach().clone()
    dist.all_reduce(global_den)
    (num * dist.get_world_size() / global_den).backward()
    optimizer.step()
for (name, expected), (_, actual) in zip(
    flat.named_parameters(), partitioned.named_parameters()
):
    torch.testing.assert_close(
        actual.grad.full_tensor(),
        expected.grad.full_tensor(),
        atol=2e-5,
        rtol=2e-4,
        msg=name,
    )
    torch.testing.assert_close(
        actual.full_tensor(), expected.full_tensor(), atol=2e-5, rtol=2e-4, msg=name
    )
print(
    f"FSDP2 rank={rank}: loss-normalized gradients and optimizer step PASS", flush=True
)
dist.destroy_process_group()
