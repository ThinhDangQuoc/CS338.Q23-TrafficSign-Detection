"""CSV report export for detection history."""

from __future__ import annotations

import csv
import os
from typing import Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "static", "outputs", "report.csv")


def export_csv(rows: list[dict[str, Any]], output_path: str = REPORT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "id",
        "sign_name",
        "confidence",
        "warning_message",
        "snapshot_path",
        "source_type",
        "detected_at",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return output_path
