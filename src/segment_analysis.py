from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.core.config import DEFAULT_THRESHOLD, MODEL_PATH, REPORTS_DIR
from src.features import extract_target, load_training_data, prepare_features
from src.model_io import load_model


def bucket_validation_frame(data_path: str, random_state: int = 42) -> pd.DataFrame:
    data = load_training_data(data_path)
    X = prepare_features(data)
    y = extract_target(data)
    _, X_valid, _, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    frame = X_valid.copy()
    frame["target"] = y_valid.to_numpy()
    frame["age_bucket"] = pd.cut(
        frame["age"], bins=[17, 35, 50, 65, 100], labels=["18-35", "36-50", "51-65", "66+"]
    )
    frame["income_bucket"] = pd.qcut(
        frame["MonthlyIncome"], q=4, labels=["income_q1", "income_q2", "income_q3", "income_q4"], duplicates="drop"
    )
    frame["late_payment_bucket"] = pd.cut(
        frame["TotalLatePayments"], bins=[-1, 0, 2, 99], labels=["none", "1-2", "3+"]
    )
    return frame


def analyze_segments(
    data_path: str,
    model_path: Path = MODEL_PATH,
    output_dir: Path = REPORTS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = bucket_validation_frame(data_path)
    model = load_model(model_path)
    feature_columns = [column for column in frame.columns if column not in {"target", "age_bucket", "income_bucket", "late_payment_bucket"}]
    frame["probability"] = model.predict_proba(frame[feature_columns])[:, 1]
    frame["prediction"] = (frame["probability"] >= threshold).astype(int)

    rows = []
    for segment_name in ["age_bucket", "income_bucket", "late_payment_bucket"]:
        for segment_value, group in frame.groupby(segment_name, observed=True):
            tn, fp, fn, tp = confusion_matrix(group["target"], group["prediction"], labels=[0, 1]).ravel()
            rows.append(
                {
                    "segment": segment_name,
                    "bucket": str(segment_value),
                    "rows": int(len(group)),
                    "default_rate": float(group["target"].mean()),
                    "avg_probability": float(group["probability"].mean()),
                    "precision": precision_score(group["target"], group["prediction"], zero_division=0),
                    "recall": recall_score(group["target"], group["prediction"], zero_division=0),
                    "f1": f1_score(group["target"], group["prediction"], zero_division=0),
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["segment", "bucket"])
    metrics.to_csv(output_dir / "segment_metrics.csv", index=False)
    write_segment_plot(metrics, output_dir)
    write_segment_summary(metrics, output_dir)
    return metrics


def write_segment_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    plot_data = metrics.copy()
    plot_data["label"] = plot_data["segment"] + ": " + plot_data["bucket"]
    plot_data = plot_data.sort_values("recall")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_data["label"], plot_data["recall"], color="#c45131")
    plt.xlabel("Default recall")
    plt.title("Model recall by borrower segment")
    plt.tight_layout()
    plt.savefig(output_dir / "segment_recall.png", dpi=160)
    plt.close()


def write_segment_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    weakest = metrics.sort_values("recall").head(3)
    risky = metrics.sort_values("default_rate", ascending=False).head(3)
    lines = [
        "# Segment Error Analysis",
        "",
        "Lower recall means more observed defaults are missed inside that borrower group.",
        "",
        "## Weakest Recall Segments",
    ]
    for _, row in weakest.iterrows():
        lines.append(
            f"- {row['segment']}={row['bucket']}: recall={row['recall']:.3f}, "
            f"FN={int(row['false_negative'])}, default_rate={row['default_rate']:.3f}"
        )
    lines.extend(["", "## Highest Observed Risk Segments"])
    for _, row in risky.iterrows():
        lines.append(
            f"- {row['segment']}={row['bucket']}: default_rate={row['default_rate']:.3f}, "
            f"avg_probability={row['avg_probability']:.3f}"
        )
    (output_dir / "segment_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segment-level error analysis.")
    parser.add_argument("--data", required=True, help="Path to cs-training.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze_segments(
        args.data,
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
