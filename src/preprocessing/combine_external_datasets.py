"""
Utilities to merge external hate-speech datasets into one binary dataset.

This script normalizes the two user-provided CSVs into a single training file:
- `combined_hate_speech_dataset 2.csv`
- `Indo-HateSpeech_Dataset.csv`

Output columns are compatible with `train.py`, which expects a text column and
any column containing `label` in its name.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text_key(text: str) -> str:
    """Create a simple normalized key for cross-dataset deduplication."""
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def map_combined_label(raw_label: str) -> int:
    value = str(raw_label).strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Unexpected label in combined dataset: {raw_label!r}")
    return int(value)


def map_indo_label(raw_label: str) -> int:
    """
    Indo-HateSpeech uses:
    - HS0: no hate
    - HS1: hate
    - HSN: extreme hate

    For binary hate-speech training, both HS1 and HSN map to 1.
    """
    value = str(raw_label).strip().strip("'").upper()
    if value == "HS0":
        return 0
    if value in {"HS1", "HSN"}:
        return 1
    raise ValueError(f"Unexpected label in Indo dataset: {raw_label!r}")


def merge_datasets(combined_path: Path, indo_path: Path, output_path: Path, report_path: Path) -> dict:
    rows_out = []
    seen = set()
    report = {
        "inputs": {
            "combined": str(combined_path),
            "indo": str(indo_path),
        },
        "datasets": {},
    }

    specs = [
        {
            "name": "combined_hate_speech_dataset_2",
            "path": combined_path,
            "text_col": "text",
            "label_col": "hate_label",
            "language_col": "language",
            "map_label": map_combined_label,
        },
        {
            "name": "indo_hatespeech_dataset",
            "path": indo_path,
            "text_col": "Comment",
            "label_col": "Label",
            "language_col": None,
            "map_label": map_indo_label,
        },
    ]

    for spec in specs:
        stats = Counter()
        label_counts = Counter()

        with spec["path"].open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stats["total_rows"] += 1

                text = (row.get(spec["text_col"]) or "").strip()
                if not text:
                    stats["dropped_empty_text"] += 1
                    continue

                norm_key = normalize_text_key(text)
                if not norm_key:
                    stats["dropped_empty_text"] += 1
                    continue
                if norm_key in seen:
                    stats["dropped_duplicates"] += 1
                    continue

                label = spec["map_label"](row.get(spec["label_col"], ""))
                seen.add(norm_key)
                label_counts[str(label)] += 1
                stats["kept_rows"] += 1

                rows_out.append(
                    {
                        "text": text,
                        "label": label,
                        "source_dataset": spec["name"],
                        "original_label": row.get(spec["label_col"], ""),
                        "language_hint": (row.get(spec["language_col"]) or "").strip()
                        if spec["language_col"]
                        else "",
                    }
                )

        report["datasets"][spec["name"]] = {
            **dict(stats),
            "label_counts": dict(label_counts),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "text",
                "label",
                "source_dataset",
                "original_label",
                "language_hint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    merged_counts = Counter(str(row["label"]) for row in rows_out)
    report["merged"] = {
        "rows": len(rows_out),
        "label_counts": dict(merged_counts),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two hate-speech CSVs into one binary dataset.")
    parser.add_argument(
        "--combined",
        default="/Users/chitraksh.verma/Downloads/combined_hate_speech_dataset 2.csv",
        help="Path to combined_hate_speech_dataset 2.csv",
    )
    parser.add_argument(
        "--indo",
        default="/Users/chitraksh.verma/Downloads/Indo-HateSpeech_Dataset.csv",
        help="Path to Indo-HateSpeech_Dataset.csv",
    )
    parser.add_argument(
        "--output",
        default="data/raw/merged_hate_speech_training.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--report",
        default="data/raw/merged_hate_speech_training_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    report = merge_datasets(
        combined_path=Path(args.combined),
        indo_path=Path(args.indo),
        output_path=Path(args.output),
        report_path=Path(args.report),
    )

    print("Merged dataset created.")
    print(f"Rows: {report['merged']['rows']}")
    print(f"Label counts: {report['merged']['label_counts']}")
    print(f"CSV: {Path(args.output).resolve()}")
    print(f"Report: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
