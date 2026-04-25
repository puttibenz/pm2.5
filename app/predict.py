# Pipeline ครบ: โหลดข้อมูล → Feature Engineering → Load Model → Predict → Save
# รันทุกวันหลัง fetch_daily.py เสร็จ
# ═══════════════════════════════════════════════════════════════

import joblib
import json 
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Config

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / 'artifacts'
DATA_DIR = REPO_ROOT / 'data'
OUTPUT_DIR = REPO_ROOT / 'predictions'
OUTPUT_DIR.mkdir(exist_ok=True)

PROVINCES = [
    'Chiang Mai', 'Chiang Rai', 'Lampang', 'Lamphun',
    'Mae Hong Son', 'Nan', 'Phayao', 'Phrae'
]

METO_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m"
]

# Load Artifacts
def load_artifacts():
    required = [
        'xgboost_pm25.pkl', 'scaler.pkl', 'feature_list.json'
    ]
    missing = [f for f in required if not (ARTIFACT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"ไม่พบ artifact: {missing}\n"
            f"รัน training script ก่อนเพื่อ save model"
        )

    model = joblib.load(ARTIFACT_DIR / 'xgboost_pm25.pkl')
    scaler = joblib.load(ARTIFACT_DIR / 'scaler.pkl')
    with open(ARTIFACT_DIR / 'feature_list.json', encoding='utf-8') as f:
        feature_list = json.load(f)
    return model, scaler, feature_list

# Load Data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    โหลด meteo และ hotspot
    ดึงย้อนหลัง 5 วัน เผื่อ lag_72h ครบ + buffer
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=5)

    meteo_candidates = [
        DATA_DIR / "raw" / "openmeteo_all_provinces.csv",
        DATA_DIR / "raw" / "openmeteo_all_provinces_2023-.csv",
        REPO_ROOT / "openmeteo_all_provinces_2023-.csv",
    ]
    meteo_path = next((path for path in meteo_candidates if path.exists()), None)
    if meteo_path is None:
        raise FileNotFoundError("ไม่พบไฟล์ meteo ที่ต้องใช้สำหรับ predict")

    meteo = pd.read_csv(meteo_path, parse_dates=['Datetime'])
    meteo = meteo[meteo['Datetime'] >= cutoff].copy()
    meteo = meteo.sort_values(['Province', 'Datetime']).reset_index(drop=True)
    print(f"  Meteo rows (5d): {len(meteo):,}")

    hotspot_candidates = [
        DATA_DIR / "processed" / "firms_daily_by_province.csv",
        DATA_DIR / "raw" / "firms_north_viirs.csv",
    ]
    hotspot_path = next((path for path in hotspot_candidates if path.exists()), None)
    if hotspot_path is not None:
        hotspot = pd.read_csv(hotspot_path, parse_dates=['date'])
        hotspot = hotspot[hotspot['date'] >= cutoff.normalize()].copy()
    else:
        print("  WARN: ไม่พบ firms_daily_by_province.csv — ใช้ hotspot=0")
        dates = pd.date_range(cutoff.date(), pd.Timestamp.now().date(), freq='D')
        hotspot = pd.MultiIndex.from_product(
            [dates, PROVINCES], names=['date', 'Province']
        ).to_frame(index=False)
        hotspot[["hotspot_count", "frp_sum", "frp_mean"]] = 0

    print(f'Hotspot rows (5d): {len(hotspot):,}')
    return meteo, hotspot
        
# Feature Engineering
def build_features(meteo: pd.DataFrame, hotspot: pd.DataFrame) -> pd.DataFrame:
    """สร้าง features เหมือน merge_and_features.py ทุกอย่าง"""
 
    # merge hotspot รายวัน เข้า meteo รายชั่วโมง
    meteo["date"] = meteo["Datetime"].dt.normalize()
    df = meteo.merge(hotspot, on=["date", "Province"], how="left")
    df[["hotspot_count", "frp_sum", "frp_mean"]] = (
        df[["hotspot_count", "frp_sum", "frp_mean"]].fillna(0)
    )
 
    frames = []
    for prov in PROVINCES:
        p = df[df["Province"] == prov].copy().sort_values("Datetime")
 
        # lag PM25
        for h in [1, 3, 6, 12, 24, 48, 72]:
            p[f"pm25_lag_{h}h"] = p["PM25"].shift(h)
 
        # rolling PM25
        for window in [6, 12, 24, 72]:
            p[f"pm25_roll_mean_{window}h"] = p["PM25"].shift(1).rolling(window).mean()
            p[f"pm25_roll_max_{window}h"]  = p["PM25"].shift(1).rolling(window).max()
 
        # rolling hotspot
        p["hotspot_roll3d"] = p["hotspot_count"].shift(24).rolling(24 * 3).mean()
        p["hotspot_roll7d"] = p["hotspot_count"].shift(24).rolling(24 * 7).mean()
        p["frp_roll3d"]     = p["frp_sum"].shift(24).rolling(24 * 3).mean()
 
        # wind direction → sin/cos (circular encoding)
        wd_rad        = np.deg2rad(p["wind_direction_10m"])
        p["wind_sin"] = np.sin(wd_rad)
        p["wind_cos"] = np.cos(wd_rad)
 
        # time features
        p["hour"]      = p["Datetime"].dt.hour
        p["month"]     = p["Datetime"].dt.month
        p["dayofweek"] = p["Datetime"].dt.dayofweek
        p["dayofyear"] = p["Datetime"].dt.dayofyear
 
        # cyclical time encoding
        p["hour_sin"]      = np.sin(2 * np.pi * p["hour"] / 24)
        p["hour_cos"]      = np.cos(2 * np.pi * p["hour"] / 24)
        p["month_sin"]     = np.sin(2 * np.pi * p["month"] / 12)
        p["month_cos"]     = np.cos(2 * np.pi * p["month"] / 12)
        p["dayofyear_sin"] = np.sin(2 * np.pi * p["dayofyear"] / 365)
        p["dayofyear_cos"] = np.cos(2 * np.pi * p["dayofyear"] / 365)
 
        frames.append(p)
 
    return pd.concat(frames, ignore_index=True)

# Predict

def run_predict(df: pd.DataFrame, model, scaler, feature_list: list) -> pd.DataFrame:
    """
    เลือกแถวล่าสุดของแต่ละจังหวัด แล้ว predict
    คืน DataFrame ผลลัพธ์พร้อม metadata
    """
    # Select latest row per province
    latest = (
        df.dropna(subset=['pm25_lag_72h'])
        .sort_values('Datetime')
        .groupby('Province')
        .last()
        .reset_index()
    )

    if latest.empty:
        raise ValueError(
            "ไม่มีข้อมูลพอสำหรับ predict\n"
            "ตรวจสอบว่าดึงข้อมูลย้อนหลังอย่างน้อย 4 วัน"
        )

    # จัดลำดับ column ให้ตรงกับตอน train
    missing_features = [f for f in feature_list if f not in latest.columns]
    if missing_features:
        raise ValueError(f"Features หายไป: {missing_features}")
        
    X = latest[feature_list].copy()

    # Check for NaN
    nan_mask = X.isna().any(axis=1)
    if nan_mask.any():
        bad_provs = latest.loc[nan_mask, "Province"].tolist()
        print(f"  WARN: NaN ใน {bad_provs} — ข้ามจังหวัดเหล่านี้")
        latest = latest[~nan_mask].copy()
        X      = X[~nan_mask].copy()
        
    # Scale (transform)
    X_scaled = scaler.transform(X)

    # Predict ทีละ horizon
    results = latest.copy()[["Province", "Datetime"]].copy()
    results.rename(columns={'PM25': 'pm25_actual', 'Datetime': 'predict_datetime'}, inplace=True)

    for horizon in [1, 3, 6, 12, 24, 48, 72]:
        col = f'target_pm25_{horizon}h'
        if hasattr(model, col):
            # multi-output model
            results[f'pred_{horizon}h'] = model.predict(X_scaled)
        else:
            # single-output: ใช้ model เดิม (ควร train แยกต่อ horizon)
            results[f'pred_{horizon}h'] = model.predict(X_scaled)
    
    results['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return results

# Save Predictions

def save_predictions(results: pd.DataFrame):
    """
    บันทึก 2 ที่:
    - predictions_latest.csv  : เขียนทับทุกวัน (Streamlit อ่านไฟล์นี้)
    - predictions_history.csv : ต่อท้ายสะสม (สำหรับ monitor accuracy)
    """
    # latest (overwrite)
    latest_path = OUTPUT_DIR / "predictions_latest.csv"
    results.to_csv(latest_path, index=False)
    print(f"  Saved → {latest_path}")

    # history (append)
    history_path = OUTPUT_DIR / "predictions_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, results], ignore_index=True)
    else:
        history = results
    history.to_csv(history_path, index=False)
    print(f"  Saved → {history_path} ({len(history)} rows total)")

# Main
if __name__ == "__main__":
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running predict pipeline...")
 
    print("\n1. Loading artifacts...")
    model, scaler, feature_list = load_artifacts()
 
    print("\n2. Loading data...")
    meteo, hotspot = load_data()
 
    print("\n3. Building features...")
    df = build_features(meteo, hotspot)
    print(f"  Feature rows: {len(df):,}, columns: {df.shape[1]}")
 
    print("\n4. Predicting...")
    results = run_predict(df, model, scaler, feature_list)
 
    print("\n5. Saving predictions...")
    save_predictions(results)
 
    print("\nResult preview:")
    print(results[["Province", "pred_24h", "pred_48h", "pred_72h"]].to_string(index=False))
    print("\nDone.")       
