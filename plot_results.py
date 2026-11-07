"""
evaluate_models.py 및 tune_hybrid.py에서 생성된 CSV 파일(model_metrics.csv, tuning_metrics.csv)을 읽어
모델 성능 비교 차트(PNG 이미지 3종)를 자동 생성하는 모듈입니다.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def numeric(row, name):
    return float(row[name]) if row.get(name, "") else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = read_rows(args.results / "model_metrics.csv")
    if not rows:
        raise RuntimeError("model_metrics.csv is empty")
    names = [row["model"] for row in rows]
    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rating_rows = [row for row in rows if numeric(row, "mae") is not None]
    if rating_rows:
        labels = [row["model"] for row in rating_rows]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([value - 0.18 for value in x], [numeric(row, "mae") for row in rating_rows], width=0.36, label="MAE")
        ax.bar([value + 0.18 for value in x], [numeric(row, "rmse") for row in rating_rows], width=0.36, label="RMSE")
        ax.set_xticks(list(x), labels); ax.set_ylabel("Error (lower is better)"); ax.set_title("Held-out rating prediction")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / "rating_error_comparison.png", dpi=160); plt.close(fig)

    precision_columns = sorted(column for column in rows[0] if column.startswith("precision_at_"))
    if precision_columns:
        ks = [int(column.rsplit("_", 1)[1]) for column in precision_columns]
        fig, ax = plt.subplots(figsize=(7, 4))
        for row in rows:
            values = [numeric(row, column) for column in precision_columns]
            if all(value is not None for value in values):
                ax.plot(ks, values, marker="o", label=row["model"])
        ax.set_xticks(ks); ax.set_xlabel("K"); ax.set_ylabel("Precision@K"); ax.set_title("Top-K recommendation comparison")
        ax.legend(); fig.tight_layout(); fig.savefig(figures / "precision_at_k_comparison.png", dpi=160); plt.close(fig)

    tuning = args.results / "tuning_metrics.csv"
    if tuning.exists():
        rows = read_rows(tuning)
        rows.sort(key=lambda row: float(row["alpha_content"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([float(row["alpha_content"]) for row in rows], [float(row["ndcg_at_k"]) for row in rows], marker="o")
        ax.set_xlabel("Content weight (alpha)"); ax.set_ylabel("Validation NDCG@K"); ax.set_title("MovieLens validation: Hybrid weight selection")
        fig.tight_layout(); fig.savefig(figures / "hybrid_alpha_tuning.png", dpi=160); plt.close(fig)

    print(f"Saved figures to {figures.resolve()}")


if __name__ == "__main__":
    main()
