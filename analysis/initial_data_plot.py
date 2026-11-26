from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
"""
TODO:

1-Histogram of classes
2-Histogram of sentence length for each class
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create exploratory plots for the PPGDPR dataset." )
    parser.add_argument("--input-path", type=Path, required=True, help="Path to the JSON dataset.", )
    parser.add_argument( "--output-dir", type=Path, required=True, help="Folder where plots will be saved.",)
    return parser.parse_args()

def load_records(input_path: Path) -> pd.DataFrame:
    with input_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    records = payload["records"]

    if not records:
        raise ValueError("The JSON file contains no records.")

    df = pd.DataFrame(records)

    required_columns = {"sentence", "label", "label_name"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["label"] = pd.to_numeric(df["label"])
    df["sentence_length_words"] = df["sentence"].fillna("").str.split().str.len()

    return df

def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    class_counts = (
        df.groupby(["label", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("label")
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=class_counts,
        x="label_name",
        y="count",
        hue="label_name",
        palette="viridis",
        legend=False,
    )

    ax.set_title("PPGDPR Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Sentences")
    plt.xticks(rotation=35, ha="right")

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()

def plot_sentence_lengths(df: pd.DataFrame, output_dir: Path) -> None:
    label_order = (
        df[["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")["label_name"]
        .tolist()
    )

    df["label_name"] = pd.Categorical(
        df["label_name"],
        categories=label_order,
        ordered=True,
    )

    grid = sns.FacetGrid(
        df,
        col="label_name",
        col_wrap=3,
        height=3.2,
        sharex=False,
        sharey=False,
    )

    grid.map_dataframe(
        sns.histplot,
        x="sentence_length_words",
        bins=20,
        color="#4C72B0",
        edgecolor="white",
    )

    grid.set_axis_labels("Sentence length (words)", "Number of sentences")
    grid.set_titles("{col_name}")
    grid.fig.suptitle(
        "Sentence-Length Distribution for Each PPGDPR Class",
        y=1.02,
        fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "sentence_lengths_by_class.png", dpi=300)
    plt.close()

def main() -> None:
    args = parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(args.input_path)

    plot_class_distribution(df, args.output_dir)
    plot_sentence_lengths(df, args.output_dir)

    print(f"Created plots in: {args.output_dir}")
    print(f"Total records plotted: {len(df)}")

if __name__ == "__main__":
    main()