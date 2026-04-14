import numpy as np
import torch
from pathlib import Path


class EarlyStopping:
    """Stop training when val F1 does not improve for `patience` epochs."""

    def __init__(self, patience: int = 10, checkpoint_path: str = "checkpoint.pth", verbose: bool = True):
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose
        self.counter = 0
        self.best_f1 = -np.inf
        self.early_stop = False

    def __call__(self, val_f1: float, model: torch.nn.Module, optimizer, epoch: int) -> None:
        if val_f1 > self.best_f1:
            if self.verbose:
                print(f"  Val F1 improved ({self.best_f1:.4f} → {val_f1:.4f}). Saving checkpoint.")
            self.best_f1 = val_f1
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "val_f1": val_f1},
                self.checkpoint_path,
            )
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best F1={self.best_f1:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
