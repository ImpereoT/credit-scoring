from fastapi.testclient import TestClient

from src.main import app


PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.30,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}


def test_explain_endpoint_returns_local_drivers():
    with TestClient(app) as client:
        response = client.post("/explain", json=PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert "top_positive_risk_drivers" in data
    assert "top_negative_risk_drivers" in data
    assert "per_feature_contribution" in data
    assert len(data["per_feature_contribution"]) == 15


def test_metrics_endpoint_counts_requests_and_predictions():
    with TestClient(app) as client:
        client.post("/predict", json=PAYLOAD)
        response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["request_count"] >= 2
    assert data["prediction_count"] >= 1
    assert data["errors_count"] >= 0


def test_model_info_endpoint_returns_model_metadata():
    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "GradientBoostingClassifier"
    assert data["threshold"] == 0.23
    assert data["feature_count"] == 15
