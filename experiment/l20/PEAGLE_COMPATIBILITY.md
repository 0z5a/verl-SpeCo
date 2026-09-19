# P-EAGLE serving contract

SpeCo baseline: `18dd7094c35d61a1710a73e8b3bd9630d0d0ffb3`.
Runtime: vLLM 0.29.0; installed source hashes in
`evidence/l20-20260919/peagle-runtime-registry.json`.

| Contract | Observed implementation | Validation |
|---|---|---|
| Architecture | SpeCo exports `LlamaForCausalLMPeagle` / `llama_peagle`. vLLM registers `PEagleDraftModel` and `PeagleLlamaForCausalLM` against `Eagle3LlamaForCausalLM`, but not the exported name. | Registry mismatch confirmed; adapter not enabled. |
| Layers | Both use a first attention projection of width 2H, followed by H-wide layers. SpeCo writes both draft layer count and `num_hidden_layers`. | Source comparison only. |
| Parameters | vLLM packs q/k/v and gate/up, prefixes backbone names with `model.`, separately consumes `mask_hidden` and d2t. | Complete checkpoint consumption not tested. |
| Mask hidden | SpeCo trains `[1,1,aux_width]`; vLLM reshapes to `[1,aux_width]` and requires it with parallel drafting. | Shape agreement only. |
| Vocabulary | SpeCo d2t stores offsets; vLLM computes target IDs as draft IDs plus offsets. vLLM skips t2d during loading. | Mapping semantics agree in source; reduced-vocab logits not tested. |
| Hidden features | Both concatenate auxiliary layers before fc, optionally normalizing each chunk. Target layer IDs and collection order must match the proposer. | Numerical parity not tested. |
| Normalization | SpeCo computes logits from final RMSNorm; vLLM returns normalized hidden states for its head. | Numerical parity not tested. |
| Positions / KV / verifier | SpeCo training mask combines causal depth-0 context with each anchor's depth chain. Runtime proposer semantics must match this at inference. | Not tested. |
| TP / graph / updates | Existing EAGLE3 runtime paths do not establish P-EAGLE checkpoint, shard, graph or revision correctness. | Not tested. |

The unsupported serving guard remains. The next meaningful test is a tiny,
complete checkpoint loaded into the registered runtime class, followed by
fixed-input draft-logit parity. Changing the architecture name alone does not
complete this validation.

## Tiny runtime probes

Deterministic 4-layer/64-hidden Llama target and 2-layer P-EAGLE, vocabulary
256, BF16, TP1, eager, two fixed token prompts and 16 generated tokens:

| Configuration | Result | Speed comparison |
|---|---|---|
| Target only | Generation completed, exit 0; runtime emitted an engine-shutdown warning. | Not a performance experiment. |
| Original `llama_peagle` export | Rejected by Transformers configuration registry before loading. | Not applicable. |
| `llama` + `PEagleDraftModel` | Initially recognized, then EAGLEConfig rewrote the name to unsupported `Eagle3PEagleDraftModel`. | Not applicable. |
| `llama` + `Eagle3LlamaForCausalLM` | Exit 0; 25 logical tensors match exactly, 2×16 generated tokens equal the target-only baseline. | Pending draft-logit correctness. |

Original and first mapped failure logs are retained. The follow-up captures
the first real draft forward inputs, outputs, logits and attention tensors
for a numerical oracle. A completed generation alone cannot establish draft
logit parity. The random fixture is an interface probe, not model-quality or
full-model E2E evidence. Runtime-only deeper `hidden_norm` parameters are unused
in that runtime's forward; all consumed checkpoint tensors still require exact
load comparison.

The initial prefill comparison failed (max logit error 0.30176, argmax mismatch).
Its post-forward hook copied the input after the runtime could mutate the residual
in place. That capture is not a valid fixed-input oracle. The hook now copies
inputs before forward; a new GPU capture is required before judging parity.
