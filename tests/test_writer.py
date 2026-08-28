from typing import Any

from nesso.data.writer import NessoCallbackWriter


def test_callback_writer_streams_prediction_and_batch() -> None:
    received: list[tuple[dict[str, Any], dict[str, Any]]] = []
    prediction = {"record_id": "example", "affinity_pred_value": -7.0}
    batch = {"residue_index": [[0, 1]]}

    NessoCallbackWriter(
        lambda result, inputs: received.append((result, inputs))
    ).write_on_batch_end(
        trainer=None,  # type: ignore[arg-type]
        pl_module=None,  # type: ignore[arg-type]
        prediction=prediction,
        batch_indices=[],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    assert received == [(prediction, batch)]
