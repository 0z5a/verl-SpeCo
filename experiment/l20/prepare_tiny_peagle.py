"""Create a deterministic small checkpoint for serving-interface validation."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
torch.manual_seed(7)
shared = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=256,
    eos_token_id=255,
    bos_token_id=1,
    pad_token_id=0,
    tie_word_embeddings=False,
)
target = LlamaForCausalLM(LlamaConfig(num_hidden_layers=4, **shared))
target.save_pretrained(args.output / "target")
draft = LlamaForCausalLMPeagle(
    PeagleConfig(
        num_hidden_layers=2,
        num_draft_layers=2,
        target_hidden_size=64,
        num_aux_hidden_states=3,
        draft_vocab_size=256,
        mask_token_id=254,
        **shared,
    )
)
with torch.no_grad():
    draft.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
draft.save_pretrained(args.output / "draft-original")
mapped = args.output / "draft-mapped"
shutil.copytree(args.output / "draft-original", mapped, dirs_exist_ok=True)
config_path = mapped / "config.json"
config = json.loads(config_path.read_text())
config.update(
    model_type="llama",
    architectures=["PEagleDraftModel"],
    eagle_config={"eagle_aux_hidden_state_layer_ids": [0, 1, 2]},
)
config_path.write_text(json.dumps(config, indent=2) + "\n")
eagle3 = args.output / "draft-eagle3"
shutil.copytree(mapped, eagle3, dirs_exist_ok=True)
config["architectures"] = ["Eagle3LlamaForCausalLM"]
(eagle3 / "config.json").write_text(json.dumps(config, indent=2) + "\n")
print(args.output)
