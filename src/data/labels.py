"""Evaluation-only ground truth from a local RCAEval manifest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.case import CaseLabel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/sample_manifest.csv"
SYSTEMS = {"ob": "online_boutique", "ss": "sock_shop", "tt": "train_ticket"}


class ManifestLabels:
    """Read once per provider; reject missing/ambiguous labels before evaluation."""

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST):
        path = Path(manifest_path)
        self.path = path if path.is_absolute() else PROJECT_ROOT / path
        self.frame = pd.read_csv(self.path, keep_default_na=False)
        required = {"case", "root_cause_service", "fault", "system"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for column in required:
            values = self.frame[column].astype(str)
            if values.ne(values.str.strip()).any():
                raise ValueError(f"Manifest has surrounding whitespace in {column}")
            if values.str.lower().isin(["", "unknown", "nan", "none"]).any():
                raise ValueError(f"Manifest has missing/unknown values in {column}")
        if self.frame.empty:
            raise ValueError("Manifest contains no cases")
        if self.frame["case"].duplicated().any():
            raise ValueError("Manifest contains duplicate case IDs")
        if not self.frame["system"].isin(SYSTEMS).all():
            raise ValueError("Manifest contains unsupported system codes")
        self.rows = self.frame.set_index("case")

    def row(self, case_id: str):
        if case_id not in self.rows.index:
            raise KeyError(f"No ground-truth label for {case_id!r} in {self.path}")
        return self.rows.loc[case_id]

    def label(self, case_id: str) -> CaseLabel:
        row = self.row(case_id)
        # Never turn an injected fault category into a fabricated metric annotation.
        indicator = str(row.get("root_cause_indicator", "")).strip()
        if indicator.lower() in {"", "unknown", "none", "nan"}:
            indicator = "unknown"  # Existing CaseLabel interface requires a string.
        return CaseLabel(
            case_id=case_id,
            root_cause_service=str(row["root_cause_service"]),
            fault_category=str(row["fault"]),
            indicator=indicator,
        )

    def system(self, case_id: str) -> str:
        return SYSTEMS[str(self.row(case_id)["system"])]
