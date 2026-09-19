"""Dedicated two-rank VeOmni 0.1.11 materialization and optimizer oracle."""

from copy import deepcopy
import os
from pathlib import Path

from safetensors.torch import save_file
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import init_empty_weights

from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
init_parallel_state(dp_size=dist.get_world_size(), dp_mode="fsdp2")
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
reference = PEagleTrainingModel(LlamaForCausalLMPeagle(config), num_depths=3)
checkpoint = Path("/experiment/evidence/l20-20260919/veomni-oracle-checkpoint")
if rank == 0:
    checkpoint.mkdir(exist_ok=True)
    save_file(reference.state_dict(), checkpoint / "model.safetensors")
dist.barrier()
with init_empty_weights():
    candidate = PEagleTrainingModel(LlamaForCausalLMPeagle(config), num_depths=3)
candidate = build_parallelize_model(
    candidate,
    weights_path=str(checkpoint),
    init_device="meta",
    mixed_precision=MixedPrecisionConfig(enable=False),
    enable_gradient_checkpointing=False,
    basic_modules=["PeagleFusedLayer", "PeagleVanillaLayer"],
)
reference = reference.cuda()
fully_shard(reference)
for name, parameter in candidate.named_parameters():
    torch.testing.assert_close(
        parameter.full_tensor(),
        dict(reference.named_parameters())[name].full_tensor(),
        rtol=0,
        atol=0,
        msg=name,
    )
for name, buffer in candidate.named_buffers():
    torch.testing.assert_close(
        buffer, dict(reference.named_buffers())[name], rtol=0, atol=0
    )
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
losses = []
for model in (reference, candidate):
    torch.manual_seed(41 + rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    numerator, denominator, _ = model(**deepcopy(batch))
    global_denominator = denominator.detach().clone()
    dist.all_reduce(global_denominator)
    loss = numerator * dist.get_world_size() / global_denominator
    losses.append(loss.detach())
    loss.backward()
    optimizer.step()
torch.testing.assert_close(*losses, atol=2e-5, rtol=2e-4)
for name, actual in candidate.named_parameters():
    expected = dict(reference.named_parameters())[name]
    torch.testing.assert_close(
        actual.full_tensor(), expected.full_tensor(), atol=2e-5, rtol=2e-4, msg=name
    )
    torch.testing.assert_close(
        actual.grad.full_tensor(),
        expected.grad.full_tensor(),
        atol=2e-5,
        rtol=2e-4,
        msg=name,
    )
print(
    f"VeOmni rank={rank}: checkpoint, buffers, loss, gradients, optimizer PASS",
    flush=True,
)
dist.destroy_process_group()
