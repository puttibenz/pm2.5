# app/evaluate_daily.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "predictions"
METRICS_PATH    = PREDICTIONS_DIR / "model_metrics.csv"


def evaluate_yesterday():
    """เปรียบเทียบ prediction เมื่อวานกับค่า PM25 จริงที่ดึงมาวันนี้"""
    yesterday = (datetime.now() - timedelta(days=1)).date()

    history_path = PREDICTIONS_DIR / "predictions_history.csv"
    if not history_path.exists():
        print("No prediction history found. Skipping evaluation.")
        return

    hist = pd.read_csv(history_path, parse_dates=['predict_datetime'])
    pred_yesterday = hist[hist['predict_datetime'].dt.date == yesterday]

    # ดึงค่าจริงจาก dashboard_data ที่มี PM25 จริงอยู่
    data_path = ROOT / "data" / "processed" / "dashboard_data.csv"
    if not data_path.exists():
        print("No actual data found. Skipping evaluation.")
        return

    actual = pd.read_csv(data_path, parse_dates=['Datetime'])
    actual_yesterday = actual[
        (actual['Datetime'].dt.date == yesterday) &
        (actual['is_predicted'] == False)
    ]

    if pred_yesterday.empty or actual_yesterday.empty:
        print(f"No data available for {yesterday}. Skipping.")
        return

    merged = pred_yesterday.merge(
        actual_yesterday[['Province', 'Datetime', 'PM25']],
        left_on=['Province', 'predict_datetime'],
        right_on=['Province', 'Datetime']
    )

    if merged.empty:
        print("No matching rows to evaluate.")
        return

    mae  = mean_absolute_error(merged['PM25'], merged['pred_24h'])
    rmse = mean_squared_error(merged['PM25'], merged['pred_24h'], squared=False)

    print(f"\n📊 Model Performance ({yesterday}):")
    print(f"   MAE  = {mae:.2f} µg/m³")
    print(f"   RMSE = {rmse:.2f} µg/m³")
    print(f"   N    = {len(merged)} samples")

    # บันทึก metrics ลง CSV เพื่อ track ตลอดเวลา
    row = pd.DataFrame([{
        'date':      yesterday.strftime('%Y-%m-%d'),
        'mae':       round(mae, 3),
        'rmse':      round(rmse, 3),
        'n_samples': len(merged),
    }])
    row.to_csv(
        METRICS_PATH,
        mode='a',
        header=not METRICS_PATH.exists(),
        index=False
    )
    print(f"   Saved -> {METRICS_PATH}")


if __name__ == "__main__":
    evaluate_yesterday()
