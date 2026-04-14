"""Evaluation utilities — session-level Top-K anomaly detection.

Adapted from eval_handle_topK in main.ipynb.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def eval_topk(
    normal_sessions: List[Tuple],
    anomaly_sessions: List[Tuple],
    model: torch.nn.Module,
    num_candidates: List[int],
    anomaly_rate: int = 1,
    device: torch.device = None,
) -> Dict[int, Tuple[float, float, float, float, float]]:
    """Evaluate using Top-K next-log prediction paradigm.

    A session is flagged anomalous if, in ≥ anomaly_rate windows,
    the true next log is not in the model's top-K predictions.

    Args:
        normal_sessions:  output of generate_eval on test_normal.csv
        anomaly_sessions: output of generate_eval on test_anomaly.csv
        model:            CSCLog model in eval mode
        num_candidates:   list of K values to evaluate (e.g. [1, 5])
        anomaly_rate:     minimum misses to flag a session (default 1)
        device:           torch device

    Returns:
        Dict {K: (accuracy, precision, recall, f1, avg_loss)}
    """
    if device is None:
        device = next(model.parameters()).device

    nor_hits: Dict[int, List[int]] = {k: [] for k in num_candidates}
    ano_hits: Dict[int, List[int]] = {k: [] for k in num_candidates}

    model.eval()
    with torch.no_grad():
        for sessions, hit_dict in [(normal_sessions, nor_hits), (anomaly_sessions, ano_hits)]:
            for seq, com, quan, timp, label in sessions:
                seq_t = torch.as_tensor(seq, dtype=torch.float, device=device)
                com_t = torch.as_tensor(com, dtype=torch.long, device=device)
                quan_t = torch.as_tensor(quan, dtype=torch.float, device=device)
                timp_t = torch.as_tensor(timp, dtype=torch.float, device=device)
                label_t = torch.as_tensor(label, dtype=torch.long, device=device)

                output = model(seq_t, com_t, quan_t, timp_t)
                sorted_idx = torch.argsort(output, dim=1, descending=True)

                for k in num_candidates:
                    top_k = sorted_idx[:, :k].contiguous()
                    misses = (~torch.isin(label_t, top_k)).sum().item()
                    hit_dict[k].append(1 if misses >= anomaly_rate else 0)

    results = {}
    nor_len = len(nor_hits[num_candidates[0]])
    ano_len = len(ano_hits[num_candidates[0]])
    true_labels = [0] * nor_len + [1] * ano_len

    for k in num_candidates:
        pred = nor_hits[k] + ano_hits[k]
        acc = accuracy_score(true_labels, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, pred, average="binary", zero_division=0
        )
        results[k] = (float(acc), float(prec), float(rec), float(f1), 0.0)

    return results


def run_test(
    test_normal_csv: str,
    test_anomaly_csv: str,
    checkpoint_path: str,
    templates_csv: str,
    emb_json: str,
    com_json: str,
    window_size: int,
    model_cfg,
    num_candidates: List[int],
    anomaly_rate: int = 1,
    device_str: str = "auto",
):
    """Load a checkpoint and evaluate on the test set. Prints results."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data.sequencer import generate_eval, load_resources
    from src.model import build_model

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )

    mapping, emb, cop, num_keys, emb_dim = load_resources(templates_csv, emb_json, com_json)
    com_num = len(cop)

    m = model_cfg
    model = build_model(
        variant=m.variant,
        input_size=emb_dim,
        com_num=com_num,
        ft_hid_size=m.ft_hid_size,
        lstm_hid_size=m.lstm_hid_size,
        mlp_hid_size=m.mlp_hid_size,
        gcn_hid_size=m.gcn_hid_size,
        out_hid_size=m.out_hid_size,
        alpha=m.alpha,
        ft_pattern=m.ft_pattern,
        num_layers=m.num_layers,
        num_keys=num_keys,
        drop=m.drop,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")

    test_normal = generate_eval(test_normal_csv, mapping, emb, cop, num_keys, emb_dim, window_size)
    test_anomaly = generate_eval(test_anomaly_csv, mapping, emb, cop, num_keys, emb_dim, window_size)

    res = eval_topk(test_normal, test_anomaly, model, num_candidates, anomaly_rate, device)
    print("\nTest Results:")
    print(f"{'K':<6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 42)
    for k, (acc, prec, rec, f1, _) in res.items():
        print(f"Top-{k:<3} {acc:>8.3f} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")
    return res
