"""Online block-drafter windows retain same-position token/hidden pairs."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from verl_speco.trainer.base_trainer import DrafterBaseTrainer


@pytest.mark.parametrize(
    "backend", ["dflash", "dflash2", "dspark", "domino", "eagle3", "peagle"]
)
@pytest.mark.parametrize("metadata", ["positions", "start", "tail"])
@pytest.mark.parametrize("at_end", [False, True])
def test_collection_preserves_token_hidden_alignment(backend, metadata, at_end):
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"use_logits": False}}}}
    )
    trainer.backend = SimpleNamespace(model_type=backend)
    trainer.copy_stream = None
    trainer.pad_token_id = 0
    trainer.model_config = SimpleNamespace(pad_token_id=0)
    trainer.current_rl_step = 1
    trainer.rank = 0
    trainer.buffer_version = 0
    trainer.use_data_buffer = False
    trainer.collected_data = []
    shifted = backend in {"eagle3", "peagle"}
    start = 7 if at_end or metadata == "tail" else 3
    rows = 5 - int(shifted) if metadata == "tail" else 5
    positions = torch.arange(start, start + rows)
    batch = {"input_ids": torch.arange(12).unsqueeze(0)}
    if metadata == "positions":
        batch["hidden_positions"] = positions.unsqueeze(0)
    elif metadata == "start":
        batch["hidden_position_start"] = torch.tensor([start])
    hidden = positions.float().view(1, rows, 1).repeat(1, 1, 4)

    trainer.collect_online_data(batch, hidden)

    item = trainer.collected_data[0]
    expected_rows = min(rows, 12 - start - int(shifted))
    expected_positions = torch.arange(start, start + expected_rows)
    torch.testing.assert_close(item["hidden_states"][:, 0], expected_positions.float())
    torch.testing.assert_close(
        item["input_ids"], torch.arange(start, start + expected_rows + int(shifted))
    )
    torch.testing.assert_close(item["position_ids"], expected_positions + int(shifted))
    assert item["loss_mask"].numel() == item["input_ids"].numel()
    assert trainer.buffer_version == 1
