"""Abstract base class for all OS-specific log parsers."""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseParser(ABC):
    """Parse raw log files into a normalized DataFrame.

    Expected output columns:
        EventId    (str)  — Drain template hash
        Component  (str)  — emitting subsystem
        Timestamp  (str)  — ISO 8601 datetime string
        Content    (str)  — raw log message text
        Label      (int)  — 0 = normal, 1 = anomaly
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_dir = Path(cfg.raw_dir)
        self.sequences_dir = Path(cfg.sequences_dir)

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """Return normalized DataFrame for all source files."""

    def _normalize_timestamp(self, ts_series: pd.Series) -> pd.Series:
        """Best-effort ISO 8601 conversion; keeps originals on failure."""
        return pd.to_datetime(ts_series, errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        ).fillna(ts_series.astype(str))
