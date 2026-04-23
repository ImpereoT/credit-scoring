from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.core.config import FEATURE_NAMES, REPORTS_DIR
from src.features import build_feature_vector
from src.services.prediction import decision_from_probability, risk_level


def _positive_class_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1]
    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, 1]
    return array


def explain_prediction(
    model: Any,
    values: Mapping[str, Any],
    feature_names: list[str] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    feature_names = feature_names or FEATURE_NAMES
    feature_vector = build_feature_vector(values)
    probability = float(model.predict_proba(feature_vector)[0][1])
    contributions, method = local_contributions(model, feature_vector, feature_names)
    rows = [
        {
            "feature": name,
            "value": float(feature_vector[0, idx]),
            "contribution": float(contributions[idx]),
            "direction": "increases_risk" if contributions[idx] > 0 else "decreases_risk",
        }
        for idx, name in enumerate(feature_names)
    ]
    rows = sorted(rows, key=lambda item: abs(item["contribution"]), reverse=True)
    positive = [item for item in rows if item["contribution"] > 0][:top_n]
    negative = [item for item in rows if item["contribution"] < 0][:top_n]
    rounded_probability = round(probability, 4)
    decision = decision_from_probability(probability)
    level = risk_level(probability)

    return {
        "probability": rounded_probability,
        "default_probability": rounded_probability,
        "risk_level": level,
        "decision": decision,
        "explanation_method": method,
        "summary": (
            f"The model estimates default probability at {rounded_probability:.1%}. "
            f"Decision: {decision}; risk level: {level}. The strongest drivers are "
            "listed as features that increase or decrease estimated risk."
        ),
        "interpretation_note": (
            "Positive contributions push the applicant toward higher estimated default "
            "risk; negative contributions push the estimate lower. SHAP values are "
            "model explanations, not causal proof."
        ),
        "top_positive_risk_drivers": positive,
        "top_negative_risk_drivers": negative,
        "top_feature_contributions": rows[:top_n],
        "per_feature_contribution": rows,
    }


def local_contributions(
    model: Any,
    feature_vector: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, str]:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = _positive_class_shap_values(explainer.shap_values(feature_vector))
        return values[0], "shap_tree_explainer"
    except Exception:
        importances = getattr(model, "feature_importances_", np.ones(len(feature_names)))
        centered = feature_vector[0] - np.mean(feature_vector[0])
        fallback = np.asarray(importances) * centered
        return fallback, "feature_importance_fallback"


def write_shap_artifacts(
    model: Any,
    feature_frame: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
    max_rows: int = 1000,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = feature_frame.sample(
        min(max_rows, len(feature_frame)), random_state=42
    )
    try:
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = _positive_class_shap_values(explainer.shap_values(sample))
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame(
            {"feature": sample.columns, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / "shap_global_importance.csv", index=False)

        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        method = "shap_tree_explainer"
    except Exception as error:
        import matplotlib.pyplot as plt

        importance = pd.DataFrame(
            {
                "feature": sample.columns,
                "mean_abs_shap": getattr(
                    model, "feature_importances_", np.zeros(len(sample.columns))
                ),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / "shap_global_importance.csv", index=False)
        plot_data = importance.head(15).sort_values("mean_abs_shap")
        plt.figure(figsize=(8, 6))
        plt.barh(plot_data["feature"], plot_data["mean_abs_shap"], color="#2f6f73")
        plt.xlabel("Fallback feature importance")
        plt.title("Global risk driver importance")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        method = f"feature_importance_fallback: {error}"

    sample_payload = sample.iloc[0].to_dict()
    local = explain_prediction(model, sample_payload, list(sample.columns))
    (output_dir / "shap_local_sample.json").write_text(
        json.dumps(local, indent=2), encoding="utf-8"
    )
    return {
        "method": method,
        "global_importance": str(output_dir / "shap_global_importance.csv"),
        "summary_plot": str(output_dir / "shap_summary.png"),
        "local_sample": str(output_dir / "shap_local_sample.json"),
    }
