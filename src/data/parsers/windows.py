"""Windows log parser — Loghub Windows CSV format."""
from pathlib import Path

import pandas as pd

from .base import BaseParser
from .drain import LogParser


class WindowsParser(BaseParser):
    """Parse Loghub Windows log CSV.

    Expected CSV columns:
        LineId, Date, Time, Level, Component, EventId, EventTemplate, Content

    Also supports raw .evtx files if the `evtx` package is available.
    """

    def parse(self) -> pd.DataFrame:
        frames = []
        for fname in self.cfg.source_files:
            fpath = self.raw_dir / fname
            if not fpath.exists():
                print(f"[WindowsParser] File not found, skipping: {fpath}")
                continue
            suffix = fpath.suffix.lower()
            if suffix == ".csv":
                df = self._parse_csv(fpath)
            elif suffix == ".evtx":
                df = self._parse_evtx(fpath)
            else:
                # Treat as raw text log — run Drain
                df = self._parse_raw(fpath)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No source files found in {self.raw_dir}")
        return pd.concat(frames, ignore_index=True)

    def _parse_csv(self, fpath: Path) -> pd.DataFrame:
        raw = pd.read_csv(fpath, na_filter=False)

        # If EventId column already exists from Loghub, use it; otherwise run Drain content
        if "EventId" not in raw.columns:
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
            raw = pd.read_csv(drain_out / (fpath.name + "_structured.csv"), na_filter=False)

        label_col = self.cfg.anomaly_label_column
        anomaly_levels = list(self.cfg.anomaly_levels)
        raw["Label"] = raw[label_col].isin(anomaly_levels).astype(int)

        # Combine Date + Time into timestamp
        if "Date" in raw.columns and "Time" in raw.columns:
            ts_str = raw["Date"].astype(str) + " " + raw["Time"].astype(str)
        else:
            ts_str = raw[self.cfg.timestamp_column].astype(str)

        raw["Timestamp"] = self._normalize_timestamp(ts_str)

        return pd.DataFrame(
            {
                "EventId": raw["EventId"],
                "Component": raw[self.cfg.component_column].str.strip(),
                "Timestamp": raw["Timestamp"],
                "Content": raw[self.cfg.content_column],
                "Label": raw["Label"],
            }
        )

    def _parse_evtx(self, fpath: Path) -> pd.DataFrame:
        try:
            import evtx as evtx_lib
        except ImportError:
            raise ImportError("Install the 'evtx' package to parse .evtx files: pip install evtx")

        import json
        rows = []
        with evtx_lib.PyEvtxParser(str(fpath)) as parser:
            for record in parser.records_json():
                try:
                    data = json.loads(record["data"])
                    system = data.get("Event", {}).get("System", {})
                    event_data = data.get("Event", {}).get("EventData", {}) or {}
                    row = {
                        "EventId": str(system.get("EventID", {}).get("#text", system.get("EventID", ""))),
                        "Component": str(system.get("Provider", {}).get("#attributes", {}).get("Name", "Unknown")),
                        "Timestamp": str(system.get("TimeCreated", {}).get("#attributes", {}).get("SystemTime", "")),
                        "Content": " ".join(f"{k}={v}" for k, v in event_data.items() if isinstance(v, str)),
                        "Level": str(system.get("Level", "0")),
                    }
                    rows.append(row)
                except Exception:
                    continue

        df = pd.DataFrame(rows)
        # Level 1=Critical, 2=Error, 3=Warning, 4=Info, 5=Verbose
        anomaly_levels = {"1", "2"}
        df["Label"] = df["Level"].isin(anomaly_levels).astype(int)
        df["Timestamp"] = self._normalize_timestamp(df["Timestamp"])
        return df[["EventId", "Component", "Timestamp", "Content", "Label"]]

    def _parse_raw(self, fpath: Path) -> pd.DataFrame:
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
        structured["Label"] = structured[self.cfg.anomaly_label_column].isin(list(self.cfg.anomaly_levels)).astype(int)
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
