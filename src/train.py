import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248
DRIFT_WARNING_POINTS = 0.05
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def _threshold_search(y_true, positive_proba):
    best_threshold = 0.5
    best_f1 = -1.0
    rows = []

    for step in range(10, 91, 5):
        threshold = step / 100
        preds = (positive_proba >= threshold).astype(int)
        score = float(f1_score(y_true, preds))
        rows.append({"threshold": threshold, "f1_score": score})
        if score > best_f1:
            best_threshold = threshold
            best_f1 = score

    return best_threshold, best_f1, rows


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))
        positive_ratio = float(y_train.mean())
        drift_delta = abs(positive_ratio - REFERENCE_POSITIVE_RATIO)
        drift_warning = drift_delta > DRIFT_WARNING_POINTS

        positive_proba = model.predict_proba(X_eval)[:, 1]
        best_threshold, best_threshold_f1, threshold_scores = _threshold_search(
            y_eval,
            positive_proba,
        )
        threshold_preds = (positive_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_eval, threshold_preds)
        precision, recall, _, support = precision_recall_fscore_support(
            y_eval,
            threshold_preds,
            labels=[0, 1],
            zero_division=0,
        )

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1_score", best_threshold_f1)
        mlflow.log_metric("train_positive_ratio", positive_ratio)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Best threshold: {best_threshold:.2f} | F1: {best_threshold_f1:.4f}")
        print(f"Train positive ratio: {positive_ratio:.3f}")
        if drift_warning:
            print(
                "WARNING: train positive ratio drifted more than "
                f"{DRIFT_WARNING_POINTS:.0%} from reference {REFERENCE_POSITIVE_RATIO:.1%}"
            )

        os.makedirs("outputs", exist_ok=True)
        report = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_threshold,
            "best_threshold_f1_score": best_threshold_f1,
            "threshold_scores": threshold_scores,
            "train_positive_ratio": positive_ratio,
            "drift_warning": drift_warning,
        }
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("Confusion matrix at best threshold\n")
            f.write("[[tn, fp], [fn, tp]]\n")
            f.write(f"{cm.tolist()}\n\n")
            f.write("Precision / recall by class at best threshold\n")
            f.write(
                f"class 0 thu_nhap_thap: precision={precision[0]:.4f}, "
                f"recall={recall[0]:.4f}, support={int(support[0])}\n"
            )
            f.write(
                f"class 1 thu_nhap_cao: precision={precision[1]:.4f}, "
                f"recall={recall[1]:.4f}, support={int(support[1])}\n\n"
            )
            f.write(classification_report(y_eval, threshold_preds, zero_division=0))
            f.write("\n\n")
            f.write(
                "Nhan xet: trong bai toan thu nhap cao, bo sot nguoi thu nhap cao "
                "(recall lop 1 thap) thuong dang chu y hon neu he thong duoc dung "
                "de tim ung vien can xem xet them; precision thap lam tang so ca "
                "can kiem tra thu cong.\n"
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
