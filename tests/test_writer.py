import json
from pathlib import Path

import torch
from safetensors.torch import load_file

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


def _write_example(predictions_dir: Path, **extra: object) -> NessoWriter:
    writer = NessoWriter(predictions_dir)
    writer.write_on_batch_end(
        trainer=None,  # type: ignore[arg-type]
        pl_module=None,  # type: ignore[arg-type]
        prediction={
            "record_id": "example",
            "affinity_pred_value": torch.tensor([-7.0]),
            "pocket_mask": torch.tensor([[1, 0]]),
            "exception": False,
            **extra,
        },
        batch_indices=[],
        batch={
            "asym_id": torch.tensor([[0, 1]]),
            "residue_index": torch.tensor([[4, 8]]),
            "mol_type": torch.tensor([[0, 3]]),
        },
        batch_idx=0,
        dataloader_idx=0,
    )
    return writer


def test_writer_commits_float32_embeddings_before_affinity_json(
    tmp_path: Path, monkeypatch
) -> None:
    record_dir = tmp_path / "predictions" / "example"
    committed_before_json: list[bool] = []
    write_json = NessoWriter._write_json_atomically

    def record_order(path: Path, data: dict) -> None:
        committed_before_json.append((record_dir / "embeddings.safetensors").exists())
        write_json(path, data)

    monkeypatch.setattr(
        NessoWriter, "_write_json_atomically", staticmethod(record_order)
    )

    _write_example(
        tmp_path / "predictions",
        embeddings={
            "affinity_repr1": torch.ones(4, dtype=torch.bfloat16),
            "affinity_repr2": torch.zeros(4),
        },
    )

    embeddings = load_file(record_dir / "embeddings.safetensors")
    assert committed_before_json == [True]
    assert set(embeddings) == {"affinity_repr1", "affinity_repr2"}
    assert embeddings["affinity_repr1"].dtype == torch.float32
    assert embeddings["affinity_repr1"].tolist() == [1.0] * 4
    assert "embeddings" not in json.loads((record_dir / "affinity.json").read_text())


def test_writer_omits_embeddings_file_when_not_requested(
    tmp_path: Path,
) -> None:
    _write_example(tmp_path / "predictions")

    record_dir = tmp_path / "predictions" / "example"
    assert (record_dir / "affinity.json").exists()
    assert not (record_dir / "embeddings.safetensors").exists()
