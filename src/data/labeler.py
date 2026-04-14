"""Split parsed DataFrames into train/test CSVs and build component maps.

Input:  DataFrame with columns [EventId, Component, Timestamp, Label, Content]
Output: sequences_dir/
    train_normal.csv      — normal log sessions for training
    test_normal.csv       — normal log sessions for evaluation
    test_anomaly.csv      — anomalous log sessions for evaluation
    <name>_component.json — {component_name: int_index}
    <name>_templates.csv  — derived from structured CSV (passed in)
"""
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def build_component_map(df: pd.DataFrame, output_json: str) -> dict:
    """Assign integer indices to unique component names."""
    components = sorted(df["Component"].unique())
    mapping = {c: i for i, c in enumerate(components)}
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Component map: {len(mapping)} components → {output_json}")
    return mapping


def build_sequences(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Convert a flat log DataFrame into session-grouped EventSequence rows.

    Each row has:
        EventSequence: list of (EventId, Component, Timestamp) tuples
        Label:         1 if the session contains ≥1 anomalous log, else 0
    """
    # Group consecutive logs into fixed-size sessions of window_size+1
    # (the +1 is for the prediction target in training)
    rows = []
    n = len(df)
    step = max(1, window_size)
    for start in range(0, n - window_size, step):
        chunk = df.iloc[start : start + window_size + 1]
        seq = list(zip(chunk["EventId"], chunk["Component"], chunk["Timestamp"]))
        label = int(chunk["Label"].max())
        rows.append({"EventSequence": str(seq), "Label": label})
    return pd.DataFrame(rows)


def split_and_save(
    df: pd.DataFrame,
    sequences_dir: str,
    window_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/test_(normal|anomaly) and save CSVs.

    Strategy:
    - Normal logs  → 80% train, 10% val (unused), 10% test_normal
    - Anomaly logs → all go to test_anomaly (CSCLog is trained on normal-only)
    """
    out = Path(sequences_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    normal_df = df[df["Label"] == 0].reset_index(drop=True)
    anomaly_df = df[df["Label"] == 1].reset_index(drop=True)

    # Build session sequences
    normal_seqs = build_sequences(normal_df, window_size)
    anomaly_seqs = build_sequences(anomaly_df, window_size)

    # Shuffle and split normal sessions
    idx = rng.permutation(len(normal_seqs))
    n_train = int(len(idx) * train_ratio)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    train_normal = normal_seqs.iloc[train_idx].reset_index(drop=True)
    test_normal = normal_seqs.iloc[test_idx].reset_index(drop=True)
    test_anomaly = anomaly_seqs.reset_index(drop=True)

    train_normal.to_csv(out / "train_normal.csv", index=False)
    test_normal.to_csv(out / "test_normal.csv", index=False)
    test_anomaly.to_csv(out / "test_anomaly.csv", index=False)

    print(
        f"Saved sequences to {out}:\n"
        f"  train_normal={len(train_normal)}, test_normal={len(test_normal)}, test_anomaly={len(test_anomaly)}"
    )
    return train_normal, test_normal, test_anomaly
