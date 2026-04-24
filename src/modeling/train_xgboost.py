"""เทรนโมเดล XGBoost สำหรับพยากรณ์ PM2.5"""

import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_features():
    """โหลดข้อมูล Features"""
    path = os.path.join("data", "features", "features_dataset.csv")
    df = pd.read_csv(path)
    return df


def train_model(df, target_col="pm25"):
    """เทรนโมเดล XGBoost"""
    # แยก Features และ Target
    feature_cols = [c for c in df.columns if c != target_col and c != "time"]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]

    # แบ่ง Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # สร้างและเทรนโมเดล
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ประเมินผล
    y_pred = model.predict(X_test)
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²:   {r2_score(y_test, y_pred):.4f}")

    # บันทึกโมเดล
    model_path = os.path.join("app", "saved_models", "xgboost_pm25.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"บันทึกโมเดลสำเร็จ: {model_path}")

    return model


if __name__ == "__main__":
    df = load_features()
    train_model(df)
