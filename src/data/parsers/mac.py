"""macOS log parser — Loghub Mac format."""
from pathlib import Path

import pandas as pd

from .base import BaseParser
from .drain import LogParser


class MacParser(BaseParser):
    """Parse macOS unified log (Loghub Mac format).

    Log format:
        Timestamp ThreadId ThreadId2 Datetime CallerProcess Category Message
    """

    def parse(self) -> pd.DataFrame:
        frames = []
        for fname in self.cfg.source_files:
            fpath = self.raw_dir / fname
            if not fpath.exists():
                print(f"[MacParser] File not found, skipping: {fpath}")
                continue
            frames.append(self._parse_file(fpath))
        if not frames:
            raise FileNotFoundError(f"No source files found in {self.raw_dir}")
        return pd.concat(frames, ignore_index=True)

    def _parse_file(self, fpath: Path) -> pd.DataFrame:
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

        # Severity-based anomaly: check if category contains error keywords
        label_col = self.cfg.anomaly_label_column
        anomaly_levels = [lvl.lower() for lvl in list(self.cfg.anomaly_levels)]
        structured["Label"] = structured[label_col].str.lower().isin(anomaly_levels).astype(int)
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
