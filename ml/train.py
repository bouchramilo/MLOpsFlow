# train.py
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import RandomOverSampler
from mlflow.tracking import MlflowClient

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset-diabete-processed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

EXPERIMENT_NAME = "Diabetes_Prediction_MLOps"
MODEL_NAME = "DiabetesRiskModel"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(EXPERIMENT_NAME)


# =========================
# TRAIN FUNCTION
# =========================
def train():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset introuvable. Lance preprocessing.py d'abord.")

    # -------- LOAD DATA
    data = pd.read_csv(DATA_PATH, index_col=0)
    X = data.drop(columns=["risk_category"])
    y = data["risk_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------- BALANCING
    sampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = sampler.fit_resample(
        X_train, y_train
    )

    # -------- PIPELINE
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # -------- GRID SEARCH
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        verbose=1
    )

    # -------- TRAIN
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # -------- PREDICTION
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # -------- METRICS
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # -------- SAVE LOCAL MODEL
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved locally to {MODEL_PATH}")

    try:
        with mlflow.start_run():
            # -------- LOG PARAMS
            mlflow.log_params(best_params)

            # -------- LOG METRICS
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # -------- REGISTER MODEL
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                registered_model_name=MODEL_NAME
            )
    except Exception as e:
        print(f"MLflow logging failed: {e}")


        # =========================
        # MODEL REGISTRY LOGIC
        # =========================
        client = MlflowClient()

        model_version = client.get_latest_versions(
            MODEL_NAME, stages=["None"]
        )[0].version

        client.set_model_version_tag(
            MODEL_NAME, model_version, "roc_auc", str(roc_auc)
        )

        # -------- PROMOTION RULE
        if roc_auc >= 0.90:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_version,
                stage="Staging",
                archive_existing_versions=True
            )

            # Auto Production
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_version,
                stage="Production",
                archive_existing_versions=True
            )

        print("Training terminé avec succès")
        print(f"ROC AUC : {roc_auc:.4f}")
        print(f"Model version : {model_version}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()
