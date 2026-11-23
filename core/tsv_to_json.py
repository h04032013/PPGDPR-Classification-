from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_COLUMNS = {"label", "sentence", "filename"}

LABEL_NAMES = {
    "0": "Other",
    "1": "Collect Personal Information",
    "2": "Data Retention Period",
    "3": "Data Processing Purposes",
    "4": "Contact Details",
    "5": "Right to Access",
    "6": "Right to Rectify or Erase",
    "7": "Right to Restrict of Processing",
    "8": "Right to Object to Processing",
    "9": "Right to Data Portability",
    "10": "Right to Lodge a Complaint",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description="Write one PPGDPR TSV record to a JSON file.")
    parser.add_argument( "--dataset-path", type=Path, default=None, help=("Path to PPGDPR's data.tsv. If omitted, common side-by-side PPGDPR clone locations are checked."),)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based record index to export (default: 0).",)
    parser.add_argument("--num_records", type=int, default=5, help="Number of records to read + export",)
    parser.add_argument("--output-path", type=Path, required=True, help=f"JSON output path",)
    return parser.parse_args()

def read_records(
    dataset_path: Path,
    start_index: int,
    num_records: int,
) -> list[dict[str, str]]:
    if start_index < 0:
        raise ValueError("--start-index must be zero or greater.")

    records = []

    with dataset_path.open(encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file, delimiter="\t")

        missing_columns = EXPECTED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")

        for index, row in enumerate(reader):
            if index < start_index:
                continue

            if num_records != 0 and len(records) == num_records:
                break

            records.append(
                {
                    "record_index": index,
                    "sentence": row["sentence"] or "",
                    "label": row["label"] or "",
                    "label_name": LABEL_NAMES.get(row["label"], "Unknown"),
                    "filename": row["filename"] or "",
                }
            )

    if num_records != 0 and len(records) != num_records:
        raise IndexError(
            f"Requested {num_records} records starting at {start_index}, "
         f"but only found {len(records)}."
    )

    return records


def write_json(output: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def main() -> None:
    args = parse_args()
    # Assumes ppgdpr and privacy-skills are sibling folders in CISCOSKILLS.
    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = PROJECT_ROOT.parent / "ppgdpr" / "data" / "data.tsv"

    records = read_records(
        dataset_path=dataset_path,
        start_index=args.start_index,
        num_records=args.num_records,
    )

    output = {
        "dataset": "PPGDPR",
        "dataset_path": str(dataset_path.resolve()),
        "start_index": args.start_index,
        "num_records": args.num_records,
        "records": records,
    }

    write_json(output, args.output_path)
    print(f"Wrote {len(records)} records to {args.output_path}")

if __name__ == "__main__":
    main()
