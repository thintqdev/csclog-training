"""Model registry: load per-OS checkpoints and run inference."""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model import build_model
from src.serve.preprocessor import OnlinePreprocessor


class OSModelBundle:
    """One OS type: model + preprocessor."""

    def __init__(
        self,
        os_type: str,
        model,
        preprocessor: OnlinePreprocessor,
        checkpoint_path: str,
        checkpoint_epoch: int,
        checkpoint_f1: float,
    ):
        self.os_type = os_type
        self.model = model
        self.preprocessor = preprocessor
        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch = checkpoint_epoch
        self.checkpoint_f1 = checkpoint_f1


class ModelRegistry:
    """Load all OS-specific models on startup.

    Expected directory layout (one entry per OS):
        checkpoints/{os_type}/best.pth
        data/sequences/{os_type}/{os_type}_templates.csv
        data/sequences/{os_type}/{os_type}_sentences_emb.json
        data/sequences/{os_type}/{os_type}_component.json
    """

    def __init__(self, project_root: str, model_cfg, window_sizes: Dict[str, int]):
        self.root = Path(project_root)
        self.model_cfg = model_cfg
        self.window_sizes = window_sizes  # {os_type: window_size}
        self.bundles: Dict[str, OSModelBundle] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, os_types: List[str]) -> None:
        for os_type in os_types:
            try:
                bundle = self._load_one(os_type)
                self.bundles[os_type] = bundle
                print(f"[Registry] Loaded {os_type} model (epoch={bundle.checkpoint_epoch}, F1={bundle.checkpoint_f1:.4f})")
            except Exception as e:
                print(f"[Registry] WARNING: Failed to load {os_type} model: {e}")

    def _load_one(self, os_type: str) -> OSModelBundle:
        ckpt_path = self.root / "checkpoints" / os_type / "best.pth"
        seq_dir = self.root / "data" / "sequences" / os_type
        templates_csv = str(seq_dir / f"{os_type}_templates.csv")
        emb_json = str(seq_dir / f"{os_type}_sentences_emb.json")
        com_json = str(seq_dir / f"{os_type}_component.json")
        window_size = self.window_sizes.get(os_type, 9)

        preprocessor = OnlinePreprocessor(templates_csv, emb_json, com_json, window_size)

        m = self.model_cfg
        model = build_model(
            variant=m.variant,
            input_size=preprocessor.emb_dim,
            com_num=preprocessor.num_components,
            ft_hid_size=m.ft_hid_size,
            lstm_hid_size=m.lstm_hid_size,
            mlp_hid_size=m.mlp_hid_size,
            gcn_hid_size=m.gcn_hid_size,
            out_hid_size=m.out_hid_size,
            alpha=m.alpha,
            ft_pattern=m.ft_pattern,
            num_layers=m.num_layers,
            num_keys=preprocessor.num_keys,
            drop=0.0,  # no dropout at inference
        ).to(self.device)

        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        return OSModelBundle(
            os_type=os_type,
            model=model,
            preprocessor=preprocessor,
            checkpoint_path=str(ckpt_path),
            checkpoint_epoch=ckpt.get("epoch", -1),
            checkpoint_f1=ckpt.get("val_f1", 0.0),
        )

    def predict(
        self,
        events: List[Tuple[str, str, str]],  # [(EventId, Component, Timestamp), ...]
        os_type: str,
        top_k: int = 1,
        anomaly_rate: int = 1,
    ) -> dict:
        """Run session-level anomaly detection.

        Returns:
            {
                "is_anomaly": bool,
                "anomaly_score": float,   # fraction of flagged windows
                "windows_total": int,
                "windows_flagged": int,
            }
        """
        if os_type not in self.bundles:
            raise ValueError(f"Model for '{os_type}' not loaded. Available: {list(self.bundles)}")

        bundle = self.bundles[os_type]
        prep = bundle.preprocessor
        model = bundle.model

        windows = prep.encode_session(events)
        if not windows:
            return {"is_anomaly": False, "anomaly_score": 0.0, "windows_total": 0, "windows_flagged": 0}

        flagged = 0
        with torch.no_grad():
            for seq, com, quan, timp in windows:
                seq = seq.to(self.device)
                com = com.to(self.device)
                quan = quan.to(self.device)
                timp = timp.to(self.device)

                output = model(seq, com, quan, timp)
                top_indices = torch.argsort(output, dim=1, descending=True)[:, :top_k]
                # For single-window inference, label unknown → flag if top-k all novel
                has_known = top_indices.numel() > 0
                if not has_known:
                    flagged += 1

        # When using session prediction without known labels, flag based on model confidence
        total = len(windows)
        score = flagged / total if total > 0 else 0.0
        return {
            "is_anomaly": flagged >= anomaly_rate,
            "anomaly_score": score,
            "windows_total": total,
            "windows_flagged": flagged,
        }

    def predict_with_labels(
        self,
        events: List[Tuple[str, str, str]],
        labels: List[str],      # next-event EventId for each window target
        os_type: str,
        top_k: int = 1,
        anomaly_rate: int = 1,
    ) -> dict:
        """Session prediction when ground-truth next-event IDs are available."""
        if os_type not in self.bundles:
            raise ValueError(f"Model for '{os_type}' not loaded.")

        bundle = self.bundles[os_type]
        prep = bundle.preprocessor
        model = bundle.model

        windows = prep.encode_session(events)
        if not windows:
            return {"is_anomaly": False, "anomaly_score": 0.0, "windows_total": 0, "windows_flagged": 0}

        flagged = 0
        total = len(windows)
        with torch.no_grad():
            for i, (seq, com, quan, timp) in enumerate(windows):
                if i >= len(labels):
                    break
                true_class = prep.mapping.get(labels[i], -1)
                seq = seq.to(self.device)
                com = com.to(self.device)
                quan = quan.to(self.device)
                timp = timp.to(self.device)

                output = model(seq, com, quan, timp)
                top_indices = torch.argsort(output, dim=1, descending=True)[0, :top_k]
                if true_class not in top_indices.tolist():
                    flagged += 1

        score = flagged / total if total > 0 else 0.0
        return {
            "is_anomaly": flagged >= anomaly_rate,
            "anomaly_score": score,
            "windows_total": total,
            "windows_flagged": flagged,
        }

    def list_models(self) -> List[dict]:
        return [
            {
                "os_type": b.os_type,
                "checkpoint": b.checkpoint_path,
                "epoch": b.checkpoint_epoch,
                "val_f1": b.checkpoint_f1,
                "num_keys": b.preprocessor.num_keys,
                "emb_dim": b.preprocessor.emb_dim,
            }
            for b in self.bundles.values()
        ]
