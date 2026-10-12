"""ms-marco MiniLM cross-encoder run through onnxruntime (int8, CPU)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from finqa_search.sources import CROSS_ENCODER, HubModel


class CrossEncoder:
    """Scores (query, passage) pairs; higher logits mean more relevant."""

    def __init__(
        self,
        model: HubModel = CROSS_ENCODER,
        *,
        snapshot: Path | None = None,
        max_length: int = 160,
        threads: int = 1,
        batch_size: int = 32,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        snap = snapshot or model.snapshot_dir()
        self.tokenizer = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length, strategy="only_second")
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(snap / model.onnx_file), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start : start + self.batch_size]
            enc = self.tokenizer.encode_batch([(query, p) for p in chunk])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            logits = self.session.run(
                None, {k: v for k, v in feeds.items() if k in self.input_names}
            )
            out.append(np.asarray(logits[0], dtype=np.float32).reshape(-1))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
