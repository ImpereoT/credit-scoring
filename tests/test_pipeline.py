import json

import pandas as pd

from src.core.config import DEFAULT_THRESHOLD, FEATURE_NAMES
from src.features import build_feature_vector, prepare_features
from src.model_io import load_feature_names, load_model_metadata
from src.services.prediction import decision_from_probability, risk_level


def test_prepare_features_adds_engineered_columns():
    raw = pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": 3.5,
                "age": 30,
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "DebtRatio": 0.5,
                "MonthlyIncome": 4000,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 2,
                "NumberOfDependents": 1,
            }
        ]
    )

    features = prepare_features(raw)

    assert list(features.columns) == FEATURE_NAMES
    assert features.loc[0, "RevolvingUtilizationOfUnsecuredLines"] == 1.0
    assert features.loc[0, "TotalLatePayments"] == 3
    assert features.loc[0, "HasAnyLatePayment"] == 1
    assert features.loc[0, "IsYoung"] == 1


def test_build_feature_vector_accepts_api_field_names():
    vector = build_feature_vector(
        {
            "RevolvingUtilizationOfUnsecuredLines": 0.2,
            "age": 45,
            "NumberOfTime30_59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.3,
            "MonthlyIncome": 5000,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        }
    )

    assert vector.shape == (1, 15)


def test_threshold_logic_boundaries():
    assert risk_level(0.05) == "LOW"
    assert risk_level(0.20) == "MEDIUM"
    assert risk_level(0.35) == "HIGH"
    assert decision_from_probability(DEFAULT_THRESHOLD - 0.01) == "Одобрить"
    assert decision_from_probability(DEFAULT_THRESHOLD) == "Отказать"


def test_model_artifact_metadata_loads():
    feature_names = load_feature_names()
    metadata = load_model_metadata()

    assert len(feature_names) == 15
    assert metadata["threshold"] == DEFAULT_THRESHOLD
    assert metadata["model_name"] == "GradientBoostingClassifier"
