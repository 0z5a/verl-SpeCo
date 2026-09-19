"""Check the locally built FlashAttention kernel and padding on CUDA."""

import torch
from flash_attn import flash_attn_func
from flash_attn.bert_padding import pad_input, unpad_input

for dtype in (torch.float16, torch.bfloat16):
    torch.manual_seed(3)
    q, k, v = [
        torch.randn(2, 32, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    actual = flash_attn_func(q, k, v, causal=True)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v), retain_graph=True)
    reference_grads = torch.autograd.grad(reference.sum(), (q, k, v))
    torch.testing.assert_close(actual_grads, reference_grads, atol=0.02, rtol=0.02)
    mask = torch.ones((2, 32), device="cuda", dtype=torch.bool)
    mask[0, :3] = False
    unpadded, indices, *_ = unpad_input(q.detach(), mask)
    restored = pad_input(unpadded, indices, 2, 32)
    torch.testing.assert_close(restored[mask], q.detach()[mask], atol=0, rtol=0)
    print(f"{dtype}: CUDA forward/backward and padding PASS", flush=True)
