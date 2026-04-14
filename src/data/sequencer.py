"""Encode EventSequence CSVs into PyTorch-ready tensors.

Mirrors generate_train / generate_pre from main.ipynb.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset
import dateutil.parser


def _parse_ts(s: str) -> float:
    """Return dateutil-parsed datetime; raises on failure."""
    return dateutil.parser.parse(s, yearfirst=True)


def load_resources(
    templates_csv: str,
    emb_json: str,
    com_json: str,
) -> Tuple[Dict, Dict, Dict, int, int]:
    """Load templates, embeddings, and component map.

    Returns:
        mapping   — {EventId: class_index}
        emb       — {EventId: [float, ...]}
        cop       — {component_name: int_index}
        num_keys  — number of unique event templates
        emb_dim   — embedding dimension
    """
    logTemp = pd.read_csv(templates_csv, index_col="EventId", na_filter=False)
    mapping = {eid: i for i, eid in enumerate(logTemp.index.unique())}
    emb = json.load(open(emb_json, "r"))
    cop = json.load(open(com_json, "r"))
    num_keys = len(mapping)
    emb_dim = len(next(iter(emb.values())))
    return mapping, emb, cop, num_keys, emb_dim


def generate_train(
    csv_path: str,
    mapping: Dict,
    emb: Dict,
    cop: Dict,
    num_keys: int,
    window_size: int,
) -> TensorDataset:
    """Sliding-window encoding for training (normal sessions only)."""
    data = pd.read_csv(csv_path, na_filter=False)

    inputs_enc, coms_enc, quans_enc, time_enc, labels = [], [], [], [], []

    for _, row in data.iterrows():
        seqs = eval(row["EventSequence"])  # list of (EventId, Component, Timestamp)
        n = len(seqs)
        for i in range(n - window_size):
            window = seqs[i : i + window_size]
            label = mapping[seqs[i + window_size][0]]

            # Quantity histogram
            quan = [0] * num_keys
            for ev, _, _ in window:
                quan[mapping[ev]] += 1

            start_dt = _parse_ts(window[0][2])
            inp, com, tm = [], [], []
            for ev, comp, ts in window:
                inp.append(emb[ev])
                com.append(cop[comp])
                tm.append((_parse_ts(ts) - start_dt).total_seconds())

            inputs_enc.append(inp)
            coms_enc.append(com)
            quans_enc.append(quan)
            time_enc.append(tm)
            labels.append(label)

    return TensorDataset(
        torch.as_tensor(inputs_enc, dtype=torch.float),
        torch.as_tensor(coms_enc, dtype=torch.long),
        torch.as_tensor(quans_enc, dtype=torch.float),
        torch.as_tensor(time_enc, dtype=torch.float),
        torch.as_tensor(labels, dtype=torch.long),
    )


def generate_eval(
    csv_path: str,
    mapping: Dict,
    emb: Dict,
    cop: Dict,
    num_keys: int,
    emb_dim: int,
    window_size: int,
) -> List[Tuple]:
    """Session-level encoding for evaluation.

    Returns list of (seq_windows, com_windows, quan_windows, time_windows, labels).
    Each item is one session (variable number of windows).
    """
    data = pd.read_csv(csv_path, na_filter=False)
    sessions = []

    for _, row in data.iterrows():
        seqs = eval(row["EventSequence"])
        n = len(seqs)
        inp_all, com_all, quan_all, time_all, lab_all = [], [], [], [], []

        for i in range(n - window_size):
            window = seqs[i : i + window_size]

            quan = [0] * num_keys
            for ev, _, _ in window:
                if ev in mapping:
                    quan[mapping[ev]] += 1

            start_dt = _parse_ts(window[0][2])
            seq, com, tm = [], [], []
            for ev, comp, ts in window:
                seq.append(emb.get(ev, [0.0] * emb_dim))
                com.append(cop.get(comp, 0))
                tm.append((_parse_ts(ts) - start_dt).total_seconds())

            inp_all.append(seq)
            com_all.append(com)
            quan_all.append(quan)
            time_all.append(tm)

            next_ev = seqs[i + window_size][0]
            lab_all.append(mapping.get(next_ev, -1))

        if inp_all:
            sessions.append((inp_all, com_all, quan_all, time_all, lab_all))

    return sessions
