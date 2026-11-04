import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "student_performance.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "attendance_percentage",
    "study_hours_per_day",
    "assignment_score",
    "class_participation",
    "previous_marks",
]
TARGET = "final_marks"

df = pd.read_csv(DATA_PATH)
required = FEATURES + [TARGET]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=[TARGET])
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

candidates = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=250, random_state=42, max_depth=12, min_samples_leaf=2
        )),
    ]),
}

results = []
trained = {}
for name, estimator in candidates.items():
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }
    results.append({"Model": name, **metrics})
    trained[name] = (estimator, metrics)

results_df = pd.DataFrame(results).sort_values("MAE")
best_name = str(results_df.iloc[0]["Model"])
best_model, best_metrics = trained[best_name]

# Explainable feature relationship: absolute coefficients for linear model,
# or impurity-based feature importance for random forest.
if best_name == "Random Forest":
    values = best_model.named_steps["model"].feature_importances_
else:
    values = np.abs(best_model.named_steps["model"].coef_)
feature_importance = dict(sorted(
    zip(FEATURES, [float(v) for v in values]),
    key=lambda item: item[1], reverse=True
))

bundle = {
    "model": best_model,
    "model_name": best_name,
    "features": FEATURES,
    "metrics": best_metrics,
    "feature_importance": feature_importance,
}
joblib.dump(bundle, os.path.join(MODEL_DIR, "best_model.joblib"))

print("\nModel comparison (test set):")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nSaved best model: {best_name}")
print(f"Feature ranking: {feature_importance}")
