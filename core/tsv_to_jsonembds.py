from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib import error, request

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
    parser = argparse.ArgumentParser(description="Export PPGDPR records to JSON with sentence embeddings."  )
    parser.add_argument( "--dataset-path", type=Path, default=None, help=("Path to PPGDPR's data tsv. If omitted, common side-by-side PPGDPR clone locations are checked." ), )
    parser.add_argument( "--start-index", type=int,default=0, help="Zero-based record index to begin exporting from.",)
    parser.add_argument("--num_records", type=int, default=5, help="Number of records to export Use 0 to export all remaining records.",)
    parser.add_argument("--output-path", type=Path, required=True, help="Path for the JSON output file.",)
    parser.add_argument("--embedding-model", required=True, help="Name of the locally installed Ollama embedding model.",)
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/embed", help="Ollama embeddings API endpoint.",)
    parser.add_argument("--batch-size", type=int, default=32, help="Number of sentences to embed in each Ollama request.", )

    return parser.parse_args()

def read_records(
    dataset_path: Path,
    start_index: int,
    num_records: int,
) -> list[dict[str, str | int]]:
    if start_index < 0:
        raise ValueError("--start-index must be zero or greater.")

    records: list[dict[str, str | int]] = []

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

            label = row["label"] or ""

            records.append(
                {
                    "record_index": index,
                    "sentence": row["sentence"] or "",
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, "Unknown"),
                    "filename": row["filename"] or "",
                }
            )

    if num_records != 0 and len(records) != num_records:
        raise IndexError(
            f"Requested {num_records} records starting at {start_index}, "
            f"but only found {len(records)}."
        )
    return records

def embed_sentences(
    sentences: list[str],
    model: str,
    ollama_url: str,
    batch_size: int,
) -> list[list[float]]:
    """Create one vector embedding per sentence using local Ollama."""
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    all_embeddings: list[list[float]] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]

        payload = json.dumps(
            {
                "model": model,
                "input": batch,
            }
        ).encode("utf-8")

        api_request = request.Request(
            ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(api_request, timeout=120) as response:
                result = json.load(response)

        embeddings = result.get("embeddings")

        all_embeddings.extend(embeddings)

        print(
            f"Embedded {len(all_embeddings)}/{len(sentences)} sentences...",
            flush=True,
        )

    return all_embeddings

def write_json(output: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

def main() -> None:
    args = parse_args()

    dataset_path = args.dataset_path

    records = read_records(
        dataset_path=dataset_path,
        start_index=args.start_index,
        num_records=args.num_records,
    )

    print(f"Creating embeddings for {len(records)} records using {args.embedding_model}...")

    embeddings = embed_sentences(
        sentences=[str(record["sentence"]) for record in records],
        model=args.embedding_model,
        ollama_url=args.ollama_url,
        batch_size=args.batch_size,
    )

    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding

    output = {
        "dataset": "PPGDPR",
        "dataset_path": str(dataset_path.resolve()),
        "start_index": args.start_index,
        "num_records": len(records),
        "embedding_model": args.embedding_model,
        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "records": records,
    }

    write_json(output, args.output_path)

    print(f"Wrote {len(records)} embedded records to {args.output_path}")

if __name__ == "__main__":
    main()