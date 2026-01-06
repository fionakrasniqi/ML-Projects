import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

df = pd.read_csv(r"C:\Users\Admin\Downloads\Telecom+Customer+Churn\telecom_customer_churn.csv")

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna(subset=["Total Charges"])
df["churn"] = (df["Customer Status"] == "Churned").astype(int)
df = df.drop(columns=["Customer ID", "Customer Status", "Churn Category", "Churn Reason"], errors="ignore")

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop(columns=["churn"])
y = df["churn"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipelines = {
    "log_reg": Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
    ]),
    "rf": Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
    "xgb": Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1)),
    ])
}

results = {}
best_auc = -1
best_name = None
best_model = None

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    results[name] = auc
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = pipe

print("\nModel AUC:")
for name, auc in results.items():
    print(f"- {name}: {auc:.4f}")

print(f"\nBest model: {best_name} (AUC = {best_auc:.4f})")

joblib.dump(best_model, "best_churn_model.pkl")
print("\nModel saved as best_churn_model.pkl")

first10 = X_test.iloc[:30].copy()
proba = best_model.predict_proba(first10)[:, 1]
first10["Churn Probability"] = (proba * 100).round(2).astype(str) + "%"
first10["Prediction"] = best_model.predict(first10)
print("\nPredictions for first 30 test customers:")
print(first10)

