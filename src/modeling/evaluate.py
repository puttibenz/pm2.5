"""ประเมินผลโมเดลและสร้างรายงาน"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_model(model_path=None):
    """โหลดโมเดลที่เทรนแล้ว"""
    if model_path is None:
        model_path = os.path.join("app", "saved_models", "xgboost_pm25.joblib")
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    """ประเมินผลโมเดล"""
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    print("=== ผลการประเมินโมเดล ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics, y_pred


def plot_predictions(y_test, y_pred, save_path=None):
    """พล็อตกราฟเปรียบเทียบค่าจริง vs ค่าพยากรณ์"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values, label="ค่าจริง", alpha=0.8)
    ax.plot(y_pred, label="ค่าพยากรณ์", alpha=0.8)
    ax.set_xlabel("ลำดับข้อมูล")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("เปรียบเทียบค่าจริง vs ค่าพยากรณ์ PM2.5")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"บันทึกกราฟสำเร็จ: {save_path}")
    plt.show()


if __name__ == "__main__":
    # TODO: โหลดข้อมูล test set และรันประเมินผล
    print("กรุณาเตรียม test set ก่อนรัน evaluate.py")
