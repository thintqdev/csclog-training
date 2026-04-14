"""Linux log parser — supports BGL (column-labeled) format."""
import os
from pathlib import Path

import pandas as pd

from .base import BaseParser
from .drain import LogParser


class LinuxParser(BaseParser):
    """Parse Linux BGL logs (or generic syslog).

    BGL format (space-delimited):
        Label Timestamp Date Node Time NodeRepeat Type Component Level Content
    Label: '-' = normal, anything else = anomaly type.
    """

    def parse(self) -> pd.DataFrame:
        frames = []
        for fname in self.cfg.source_files:
            fpath = self.raw_dir / fname
            if not fpath.exists():
                print(f"[LinuxParser] File not found, skipping: {fpath}")
                continue
            df = self._parse_file(fpath)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No source files found in {self.raw_dir}")
        return pd.concat(frames, ignore_index=True)

    def _parse_file(self, fpath: Path) -> pd.DataFrame:
        # Run Drain for template extraction
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

        # Build labels
        if self.cfg.anomaly_label_strategy == "column":
            col = self.cfg.anomaly_label_column
            normal_val = self.cfg.anomaly_normal_value
            structured["Label"] = (structured[col] != normal_val).astype(int)
        else:  # severity-based
            col = self.cfg.anomaly_label_column
            levels = list(self.cfg.anomaly_levels)
            structured["Label"] = structured[col].isin(levels).astype(int)

        # Normalize timestamps to ISO 8601
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
