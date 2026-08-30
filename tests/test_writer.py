import json
from pathlib import Path

import torch

from nesso.data.writer import NessoWriter


def test_writer_commits_scalars_and_binding_site_tokens(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    writer = NessoWriter(predictions_dir)
    prediction = {
        "record_id": "example",
        "affinity_pred_value": torch.tensor([-7.0]),
        "entropy_pair": torch.tensor([99.0]),
        "pocket_mask": torch.tensor([[1, 0, 1, 1]]),
        "exception": False,
    }
    batch = {
        "asym_id": torch.tensor([[0, 0, 1, 2]]),
        "residue_index": torch.tensor([[4, 5, 8, 0]]),
        "mol_type": torch.tensor([[0, 0, 0, 3]]),
    }

    writer.write_on_batch_end(
        trainer=None,  # type: ignore[arg-type]
        pl_module=None,  # type: ignore[arg-type]
        prediction=prediction,
        batch_indices=[],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    result_path = predictions_dir / "example" / "affinity.json"
    assert json.loads(result_path.read_text()) == {
        "affinity_pred_value": -7.0,
        "binding_site_tokens": [
            {"asym_id": 0, "residue_index": 4},
            {"asym_id": 1, "residue_index": 8},
        ],
    }
    assert not (tmp_path / ".example.json.tmp").exists()
