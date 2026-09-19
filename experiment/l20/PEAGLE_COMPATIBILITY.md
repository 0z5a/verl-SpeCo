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
