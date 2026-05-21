"""Reporting helpers for detection history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_summary_stats(history_df) -> dict:
    """Build basic summary statistics from a detection history DataFrame."""
    if history_df is None or len(history_df) == 0:
        return {
            "total": 0,
            "top_sign": "Chưa có dữ liệu",
            "avg_confidence": 0.0,
            "latest_detection": "Chưa có dữ liệu",
        }

    df = pd.DataFrame(history_df)
    top_sign = "Chưa có dữ liệu"
    if "sign_name" in df.columns and not df["sign_name"].empty:
        counts = df["sign_name"].value_counts()
        if not counts.empty:
            top_sign = f"{counts.index[0]} ({int(counts.iloc[0])} lần)"

    avg_confidence = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0
    latest_detection = str(df.iloc[0].get("detected_at", "Chưa có dữ liệu"))

    return {
        "total": int(len(df)),
        "top_sign": top_sign,
        "avg_confidence": round(avg_confidence, 3),
        "latest_detection": latest_detection,
    }


def export_history_csv(history_df, output_path: str = "database/detection_history.csv") -> str:
    """Export history DataFrame to CSV and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history_df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)

