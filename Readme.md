# 🏦 Credit Scoring API

> Система предсказания вероятности дефолта по кредиту на основе реальных банковских данных.  
> Включает веб-интерфейс, REST API, MLflow эксперименты, Docker-деплой и CI/CD через GitHub Actions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI](https://img.shields.io/github/actions/workflow/status/ImpereoT/credit-scoring/ci.yml?label=CI)

---

##  Результаты моделей

| Модель | ROC-AUC | PR-AUC | F1 (порог 0.23) |
|---|---|---|---|
| Logistic Regression | 0.862 | 0.389 | 0.216 |
| **GradientBoosting** ✅ | **0.868** | **0.400** | **0.448** |

**Почему PR-AUC, а не Accuracy:**  
Дисбаланс классов — только 6.7% дефолтов. Тривиальная модель «все одобрить» даёт 93.3% accuracy, но не поймает ни одного дефолта. PR-AUC честно отражает качество на положительном классе.

**Почему порог 0.23, а не 0.50:**  
Подбор по F1-score показал, что оптимальный порог — 0.23. При стандартном пороге 0.5 F1 = 0.28, при оптимальном — 0.45 (+60%).

---

##  Ключевые находки

- **TotalLatePayments** — созданный признак стал важнейшим (importance = **0.59**), превзойдя все оригинальные признаки датасета
- Молодые заёмщики (18–35 лет) показывают вдвое больший риск дефолта (~11%) против пожилых (~2.5%)
- 19.8% пропусков в `MonthlyIncome` обработаны медианной импутацией
- Выбросы в `RevolvingUtilization` (max = 50 708 при норме 0–1) обрезаны через `.clip(0, 1)`

---

##  Демо — веб-интерфейс

Три уровня риска в действии:

| 🟢 LOW — Одобрить | 🟡 MEDIUM — Одобрить | 🔴 HIGH — Отказать |
|:---:|:---:|:---:|
| ![low](screenshots/image1.png) | ![medium](screenshots/image2.png) | ![high](screenshots/image3.png) |
| 2.3% вероятность | 21.4% вероятность | 30.3% вероятность |

![MLflow](screenshots/experiments.png)

## 🗂️ Структура проекта

```
credit-scoring/
├── .github/
│   └── workflows/
│       └── ci.yml                     # CI: сборка и тест Docker образа
├── notebooks/
│   └── 01_eda_and_modeling.ipynb      # EDA, очистка, обучение, метрики
├── src/
│   ├── main.py                        # FastAPI с логированием и валидацией
│   └── frontend.html                  # Веб-интерфейс
├── models/
│   ├── model.pkl                      # Обученная модель GradientBoosting
│   └── feature_names.pkl              # Список признаков
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Быстрый старт

### Через Docker (рекомендуется)

```bash
git clone https://github.com/ImpereoT/credit-scoring.git
cd credit-scoring
docker-compose up -d
```

Веб-интерфейс: `http://localhost:8000`  
Swagger API:   `http://localhost:8000/docs`

### Локально

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
uvicorn src.main:app --reload
```

---

## 📡 API

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.30,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0
  }'
```

**Ответ:**

```json
{
  "default_probability": 0.0213,
  "risk_level": "LOW",
  "decision": "Одобрить",
  "threshold_used": 0.23
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model": "GradientBoosting",
  "threshold": 0.23,
  "features": 15
}
```

---

##  Пайплайн разработки

```
Данные (Kaggle)
      ↓
EDA + очистка данных
(пропуски, выбросы, feature engineering)
      ↓
MLflow: трекинг 3 экспериментов
(LogReg, GradBoost baseline, GradBoost deeper)
      ↓
Подбор порога по F1-score
      ↓
FastAPI (валидация, логирование, конфиг)
      ↓
Docker + docker-compose
      ↓
GitHub Actions CI (сборка + тест /health)
```

---

##  Данные

**Give Me Some Credit** — [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)

- 150 000 реальных записей банковских клиентов
- 11 признаков + 5 созданных в процессе feature engineering
- Целевая переменная: дефолт в течение 2 лет (6.7% положительного класса)

> Скачайте `cs-training.csv` и положите в `data/` для переобучения модели.

---

## 🛠️ Стек

`Python 3.11` · `scikit-learn` · `FastAPI` · `uvicorn` · `Docker` · `MLflow` · `pandas` · `seaborn` · `GitHub Actions`
