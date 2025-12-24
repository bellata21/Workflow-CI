import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
import argparse


def train_model(input_csv=None):
    current_dir = os.path.dirname(__file__)

    if input_csv is None:
        input_csv = os.path.join(current_dir, "bank_loan_preprocessing.csv")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File CSV tidak ditemukan: {input_csv}")


    mlflow.autolog()

    # load dataset
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["Personal Loan"])
    y = df["Personal Loan"]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # hyperparameter grid
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }

    base_model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # train
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # eval
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # manual logging
    mlflow.log_param("best_C", grid_search.best_params_["C"])
    mlflow.log_param("best_solver", grid_search.best_params_["solver"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # log model
    mlflow.sklearn.log_model(best_model, artifact_path="model")

    # confusion matrik
    cm_path = os.path.join(current_dir, "confusion_matrix.png")
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # kurva ROC 
    roc_path = os.path.join(current_dir, "roc_curve.png")
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close()

    print("Best Params:", grid_search.best_params_)
    print(f"Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None)
    args = parser.parse_args()

    train_model(args.input_csv)
