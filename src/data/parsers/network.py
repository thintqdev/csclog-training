"""Network device log parser — OpenStack and generic syslog formats."""
from pathlib import Path

import pandas as pd

from .base import BaseParser
from .drain import LogParser


class NetworkParser(BaseParser):
    """Parse OpenStack / network device logs.

    OpenStack format:
        LogId Date Time Pid Level Component Content
    Normal logs: openstack_normal.log (all normal)
    Abnormal logs: openstack_abnormal.log (all anomaly)
    """

    # Files whose name contains these substrings are fully anomalous
    ANOMALY_FILE_PATTERNS = ("abnormal", "attack", "attack_data")

    def parse(self) -> pd.DataFrame:
        frames = []
        for fname in self.cfg.source_files:
            fpath = self.raw_dir / fname
            if not fpath.exists():
                print(f"[NetworkParser] File not found, skipping: {fpath}")
                continue
            force_label = self._infer_forced_label(fname)
            frames.append(self._parse_file(fpath, force_label))
        if not frames:
            raise FileNotFoundError(f"No source files found in {self.raw_dir}")
        return pd.concat(frames, ignore_index=True)

    def _infer_forced_label(self, fname: str):
        """Return 1 if filename indicates all-anomaly, 0 if all-normal, None for mixed."""
        lower = fname.lower()
        if any(p in lower for p in self.ANOMALY_FILE_PATTERNS):
            return 1
        if "normal" in lower:
            return 0
        return None

    def _parse_file(self, fpath: Path, force_label) -> pd.DataFrame:
        drain_out = self.raw_dir / "drain_out"
        parser = LogParser(
            log_format=self.cfg.log_format,
            indir=str(self.raw_dir),
            outdir=str(drain_out),
            depth=self.cfg.drain.depth,
            st=self.cfg.drain.st,
            maxChild=self.cfg.drain.max_child,
            rex=list(self.cfg.drain.rex),
        )
        parser.parse(fpath.name)
        structured = pd.read_csv(drain_out / (fpath.name + "_structured.csv"), na_filter=False)

        if force_label is not None:
            structured["Label"] = force_label
        else:
            label_col = self.cfg.anomaly_label_column
            anomaly_levels = list(self.cfg.anomaly_levels)
            structured["Label"] = structured[label_col].isin(anomaly_levels).astype(int)

        structured["Timestamp"] = self._normalize_timestamp(structured[self.cfg.timestamp_column])

        return pd.DataFrame(
            {
                "EventId": structured["EventId"],
                "Component": structured[self.cfg.component_column].str.strip(),
                "Timestamp": structured["Timestamp"],
                "Content": structured[self.cfg.content_column],
                "Label": structured["Label"],
            }
        )
