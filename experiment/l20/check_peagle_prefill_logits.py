"""Compare captured runtime prefill with the original P-EAGLE forward."""

import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch.nn.attention.flex_attention import create_block_mask

from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig

root = Path("/experiment")
evidence = root / "evidence/l20-20260919"
capture = torch.load(evidence / "tiny-peagle-first-forward.pt", weights_only=True)
inputs = {key: value.cuda() for key, value in capture["inputs"].items()}
length = inputs["input_ids"].numel()
for metadata in capture["metadata"].values():
    # Only fresh single-sequence prefill: no unseen cached keys are admissible.
    assert metadata["query_start_loc"].tolist() == [0, length]
    assert metadata["seq_lens"].tolist() == [length]
torch.testing.assert_close(inputs["positions"], torch.arange(length, device="cuda"))
config = PeagleConfig.from_pretrained(root / "tiny-peagle/draft-original")
model = LlamaForCausalLMPeagle(config)
model.load_state_dict(load_file(root / "tiny-peagle/draft-original/model.safetensors"))
model = model.cuda().to(torch.bfloat16).eval()


def causal(batch, head, query, key):
    return query >= key


mask = create_block_mask(
    causal, B=1, H=None, Q_LEN=length, KV_LEN=length, device="cuda"
)
with torch.no_grad():
    hidden = model.forward_peagle(
        inputs["input_ids"].long().unsqueeze(0),
        inputs["hidden_states"].unsqueeze(0),
        inputs["positions"].unsqueeze(0),
        mask,
    )
    logits = model.compute_logits(hidden).squeeze(0).float().cpu()
expected = capture["logits"].float()
report = {
    "scope": "single-sequence causal prefill, BF16, TP1; not masked-depth decode",
    "tokens": length,
    "max_logit_error": (logits - expected).abs().max().item(),
    "argmax_equal": torch.equal(logits.argmax(-1), expected.argmax(-1)),
    "atol": 0.01,
    "rtol": 0.02,
}
(evidence / "tiny-peagle-prefill-parity.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
torch.testing.assert_close(logits, expected, atol=0.01, rtol=0.02)
assert report["argmax_equal"]
