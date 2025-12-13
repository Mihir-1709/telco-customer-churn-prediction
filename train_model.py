import pandas as pd
import joblib
import sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df = pd.read_csv("Telco-Customer-Churn.csv")

df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

features = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

X = df[features]
y = df["Churn"]

categorical = X.select_dtypes(include="object").columns.tolist()
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ]
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

param_grid = {
    "rf__max_depth": [None, 15, 25],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__max_features": ["sqrt", "log2"],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=8,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

search.fit(X, y)

best_model = search.best_estimator_

preds = best_model.predict(X)
print("Accuracy:", accuracy_score(y, preds))
print("F1 Score:", f1_score(y, preds))
print("ROC-AUC:", roc_auc_score(y, preds))

joblib.dump(best_model, "best_rf_model.pkl")
joblib.dump(features, "model_features.pkl")

print("\n✅ Training complete")
print("Saved: best_rf_model.pkl, model_features.pkl")