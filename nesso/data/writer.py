import json
import os
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from safetensors.torch import save_file

from nesso.data import const


class NessoWriter(BasePredictionWriter):
    """Writes the predict_step output of Nesso1:
    - scalars to affinity.json,
    - optionally tensors to predictions.safetensors.

    predictions.safetensors contains the following tensors:
    - z: Pairformer representation.
    - pdistogram: pdistogram tensor from the distogram head.
    - refine_mask: Sequence/complex tokens mask for refinement pass.
    - pocket_mask: Pocket token mask.
    - token_pad_mask: Original padding mask for tokens.
    - mol_type: Chain type per refined token (aligned with z).
    """

    def __init__(
        self,
        output_dir: str | Path,
        save_metadata: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")
        self.save_metadata = save_metadata
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed = 0

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: dict[str, Any],
        batch_indices: list[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        exc = prediction.get("exception")
        if exc is True or (
            isinstance(exc, torch.Tensor) and exc.numel() > 0 and bool(exc.any().item())
        ):
            self.failed += 1
            return

        rid = prediction.get("record_id")
        if not rid:
            msg = "predict_step output missing record_id"
            raise ValueError(msg)

        record_dir = self.output_dir / str(rid)
        record_dir.mkdir(parents=True, exist_ok=True)

        # Keep scalar predictions and compact residue indices in one result.
        stats: dict[str, Any] = {}
        for k, v in prediction.items():
            if k.startswith(("affinity_", "entropy_")) and k != "entropy_pair":
                if isinstance(v, torch.Tensor):
                    stats[k] = float(v.item())
                elif isinstance(v, (int, float)):
                    stats[k] = float(v)
        stats["binding_site_tokens"] = self._binding_site_tokens(
            prediction,
            batch,
        )

        stats_path = record_dir / "affinity.json"
        if stats_path.exists():
            raise FileExistsError(f"prediction already exists: {stats_path}")
        self._write_json_atomically(stats_path, stats)

        # This directory mtime marks a completely committed result generation.
        os.utime(self.output_dir)
        self._fsync_directory(self.output_dir)

        # Save additional pairwise representations if requested
        if self.save_metadata:
            if "z_full" not in prediction:
                raise ValueError("z_full not found in prediction")
            safetensors_data = {
                "z": prediction["z_full"].detach().cpu(),
                "pdistogram": prediction["pdistogram"].squeeze(0).detach().cpu(),
                "refine_mask": prediction["refine_mask_full"].detach().cpu(),
                "pocket_mask": prediction["pocket_mask_full"].detach().cpu(),
                "token_pad_mask": prediction["token_pad_mask"]
                .squeeze(0)
                .detach()
                .cpu(),
                "mol_type": prediction["mol_type_refined"].detach().cpu(),
            }
            save_file(safetensors_data, record_dir / "predictions.safetensors")

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        print(f"Number of failed examples: {self.failed}")

    @staticmethod
    def _binding_site_tokens(
        prediction: dict[str, Any],
        batch: dict[str, Any],
    ) -> list[dict[str, int]]:
        """Return selected protein residue indices from one prediction."""
        selected = prediction["pocket_mask"].bool() & (
            batch["mol_type"] == const.chain_type_ids["PROTEIN"]
        )
        asym_ids = batch["asym_id"][selected].detach().cpu().tolist()
        residue_indices = batch["residue_index"][selected].detach().cpu().tolist()
        return [
            {
                "asym_id": int(asym_id),
                "residue_index": int(residue_index),
            }
            for asym_id, residue_index in zip(
                asym_ids,
                residue_indices,
                strict=True,
            )
        ]

    @staticmethod
    def _write_json_atomically(path: Path, data: dict[str, Any]) -> None:
        """Commit JSON only after its complete contents reach local storage."""
        temporary_path = path.parent.parent.parent / (f".{path.parent.name}.json.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        NessoWriter._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Persist directory entries needed for interruption-safe resume."""
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
