"""Online preprocessing: raw log lines → model-ready tensors."""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dateutil.parser
import pandas as pd
import torch


class OnlinePreprocessor:
    """Converts a list of raw log strings into tensors for inference.

    For unknown event templates, falls back to a zero embedding (OOV handling).
    """

    def __init__(
        self,
        templates_csv: str,
        emb_json: str,
        com_json: str,
        window_size: int,
    ):
        self.window_size = window_size

        logTemp = pd.read_csv(templates_csv, index_col="EventId", na_filter=False)
        self.mapping: Dict[str, int] = {eid: i for i, eid in enumerate(logTemp.index.unique())}
        self.num_keys = len(self.mapping)

        self.emb: Dict[str, List[float]] = json.load(open(emb_json, "r"))
        self.emb_dim = len(next(iter(self.emb.values())))
        self.cop: Dict[str, int] = json.load(open(com_json, "r"))
        self.num_components = len(self.cop)
        self._zero_emb = [0.0] * self.emb_dim

    def _lookup_emb(self, event_id: str) -> List[float]:
        return self.emb.get(event_id, self._zero_emb)

    def _lookup_com(self, component: str) -> int:
        return self.cop.get(component, 0)

    def encode_window(
        self,
        events: List[Tuple[str, str, str]],  # [(EventId, Component, Timestamp), ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single window of `window_size` log events.

        Returns:
            seq   [1, W, emb_dim]   sentence embeddings
            com   [1, W]            component indices
            quan  [1, num_keys]     event histogram
            timp  [1, W]            relative timestamps (seconds)
        """
        assert len(events) == self.window_size, (
            f"Expected window of size {self.window_size}, got {len(events)}"
        )
        start_dt = dateutil.parser.parse(events[0][2], yearfirst=True)

        seq, com, tm = [], [], []
        quan = [0] * self.num_keys
        for ev, comp, ts in events:
            seq.append(self._lookup_emb(ev))
            com.append(self._lookup_com(comp))
            dt = dateutil.parser.parse(ts, yearfirst=True)
            tm.append((dt - start_dt).total_seconds())
            idx = self.mapping.get(ev)
            if idx is not None:
                quan[idx] += 1

        return (
            torch.tensor([seq], dtype=torch.float),
            torch.tensor([com], dtype=torch.long),
            torch.tensor([quan], dtype=torch.float),
            torch.tensor([tm], dtype=torch.float),
        )

    def encode_session(
        self,
        events: List[Tuple[str, str, str]],
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Encode all sliding windows in a session."""
        windows = []
        for i in range(len(events) - self.window_size):
            windows.append(self.encode_window(events[i : i + self.window_size]))
        return windows
