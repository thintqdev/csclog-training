"""Training entry point — Hydra + MLflow.

Usage:
    python src/train.py data=linux
    python src/train.py data=windows model.variant=wo_ic
    python src/train.py --multirun data=linux,windows,mac,network
"""
import os
import random
import sys
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow importing src.* from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sequencer import generate_train, generate_eval, load_resources
from src.model import build_model, EarlyStopping
from src.evaluate import eval_topk


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(cfg))

    # ── Paths ──────────────────────────────────────────────────────────────
    seq_dir = Path(cfg.data.sequences_dir)
    templates_csv = str(seq_dir / f"{cfg.data.name}_templates.csv")
    emb_json = str(seq_dir / f"{cfg.data.name}_sentences_emb.json")
    com_json = str(seq_dir / f"{cfg.data.name}_component.json")
    train_csv = str(seq_dir / "train_normal.csv")
    val_normal_csv = str(seq_dir / "test_normal.csv")
    val_anomaly_csv = str(seq_dir / "test_anomaly.csv")
    ckpt_dir = Path(cfg.train.checkpoint_dir) / cfg.data.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load resources ─────────────────────────────────────────────────────
    mapping, emb, cop, num_keys, emb_dim = load_resources(templates_csv, emb_json, com_json)
    com_num = len(cop)
    window_size = cfg.data.window_size

    print(f"num_keys={num_keys}, emb_dim={emb_dim}, com_num={com_num}")

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = generate_train(train_csv, mapping, emb, cop, num_keys, window_size)
    val_normal = generate_eval(val_normal_csv, mapping, emb, cop, num_keys, emb_dim, window_size)
    val_anomaly = generate_eval(val_anomaly_csv, mapping, emb, cop, num_keys, emb_dim, window_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    m = cfg.model
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {m.variant} — {total_params:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    early_stopping = EarlyStopping(
        patience=cfg.train.patience,
        checkpoint_path=str(ckpt_dir / "best.pth"),
    )

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("file:///" + str(Path("mlruns").resolve()))
    mlflow.set_experiment(cfg.experiment_name)

    flat_params = {
        f"data.{k}": v for k, v in OmegaConf.to_container(cfg.data, resolve=True).items()
        if not isinstance(v, (dict, list))
    }
    flat_params.update({
        f"model.{k}": v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
        if not isinstance(v, (dict, list))
    })
    flat_params.update({
        f"train.{k}": v for k, v in OmegaConf.to_container(cfg.train, resolve=True).items()
        if not isinstance(v, (dict, list))
    })

    best_f1 = 0.0
    num_candidates = list(cfg.train.num_candidates)

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(flat_params)
        mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")

        for epoch in range(cfg.train.num_epochs):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            train_losses = []
            for seq, com, quan, timp, label in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False):
                seq = seq.to(device)
                com = com.to(device)
                quan = quan.to(device)
                timp = timp.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = model(seq, com, quan, timp)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_loss = float(np.mean(train_losses))
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{cfg.train.num_epochs}] train_loss={avg_loss:.4f}")

            # ── Validate every N epochs ────────────────────────────────────
            if (epoch + 1) % cfg.train.val_every == 0:
                res = eval_topk(val_normal, val_anomaly, model, num_candidates, cfg.train.anomaly_rate, device)
                for k, (acc, prec, rec, f1, _) in res.items():
                    mlflow.log_metric(f"val_f1_top{k}", f1, step=epoch)
                    mlflow.log_metric(f"val_precision_top{k}", prec, step=epoch)
                    mlflow.log_metric(f"val_recall_top{k}", rec, step=epoch)
                    print(
                        f"  Top-{k} | Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}"
                    )

                top1_f1 = res[num_candidates[0]][3]
                early_stopping(top1_f1, model, optimizer, epoch)
                if top1_f1 > best_f1:
                    best_f1 = top1_f1

                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        mlflow.log_metric("best_val_f1", best_f1)
        mlflow.log_artifact(str(ckpt_dir / "best.pth"))
        print(f"\nBest val F1: {best_f1:.4f} — checkpoint at {ckpt_dir / 'best.pth'}")

    return best_f1


if __name__ == "__main__":
    main()
