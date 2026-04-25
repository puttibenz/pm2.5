# Pipeline ครบ: โหลดข้อมูล → Feature Engineering → Load Model → Predict → Save
# รันทุกวันหลัง fetch_daily.py เสร็จ
# ═══════════════════════════════════════════════════════════════

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / 'artifacts'
DATA_DIR     = REPO_ROOT / 'data'
OUTPUT_DIR   = REPO_ROOT / 'predictions'
OUTPUT_DIR.mkdir(exist_ok=True)

PROVINCES = [
    'Chiang Mai', 'Chiang Rai', 'Lampang', 'Lamphun',
    'Mae Hong Son', 'Nan', 'Phayao', 'Phrae'
]

# ── Feature constants (ตรงกับ modeling.ipynb) ─────────────────
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FIRE_LAGS = [24, 48, 72]
WINDOWS   = [3, 6, 12, 24, 48, 168]

# Province label encoding (LabelEncoder เรียงตามตัวอักษร จาก modeling.ipynb)
PROVINCE_LABELS = {
    'Chiang Mai': 0, 'Chiang Rai': 1, 'Lampang': 2, 'Lamphun': 3,
    'Mae Hong Son': 4, 'Nan': 5, 'Phayao': 6, 'Phrae': 7,
}

# Province target encoding (mean PM2.5 จาก train set ใน modeling.ipynb)
PROVINCE_MEAN_MAP = {
    'Chiang Mai':   21.587774223034735,
    'Chiang Rai':   19.978260207190736,
    'Lampang':      18.251279707495428,
    'Lamphun':      17.79868982327849,
    'Mae Hong Son': 12.911837294332724,
    'Nan':          18.212416209628277,
    'Phayao':       17.143692870201097,
    'Phrae':        17.761144119439365,
}

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

# ── Load Data ─────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    โหลด meteo + hotspot แล้ว merge กัน
    ดึงย้อนหลัง 10 วัน เพื่อให้ rolling_168h (7 วัน) มีข้อมูลครบ
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=10)

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
    print(f"  Meteo rows (10d): {len(meteo):,}")

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

    print(f"  Hotspot rows (10d): {len(hotspot):,}")

    # Merge hotspot (daily) เข้า meteo (hourly)
    meteo["date"] = meteo["Datetime"].dt.normalize()
    df = meteo.merge(hotspot, on=["date", "Province"], how="left")
    df[["hotspot_count", "frp_sum", "frp_mean"]] = df[["hotspot_count", "frp_sum", "frp_mean"]].fillna(0)
    return df


# ── Feature Engineering ───────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง features ตรงกับ modeling.ipynb ทุกอย่าง
    """
    frames = []

    for prov in PROVINCES:
        p = df[df["Province"] == prov].copy().sort_values("Datetime").reset_index(drop=True)

        # ── Time Features ──
        p['year']           = p['Datetime'].dt.year
        p['month']          = p['Datetime'].dt.month
        p['day']            = p['Datetime'].dt.day
        p['hour']           = p['Datetime'].dt.hour
        p['dayofyear']      = p['Datetime'].dt.dayofyear
        p['is_haze_season'] = p['month'].isin([3, 4]).astype(int)
        p['hour_sin']       = np.sin(2 * np.pi * p['hour']  / 24)
        p['hour_cos']       = np.cos(2 * np.pi * p['hour']  / 24)
        p['month_sin']      = np.sin(2 * np.pi * p['month'] / 12)
        p['month_cos']      = np.cos(2 * np.pi * p['month'] / 12)
        p['day_sin']        = np.sin(2 * np.pi * p['day']   / 365)
        p['day_cos']        = np.cos(2 * np.pi * p['day']   / 365)
        p['wind_dir_sin']   = np.sin(np.radians(p['wind_direction_10m']))
        p['wind_dir_cos']   = np.cos(np.radians(p['wind_direction_10m']))

        # ── PM2.5 Lag Features ──
        for lag in LAG_HOURS:
            p[f'pm25_lag_{lag}h'] = p['PM25'].shift(lag)

        # ── Fire Lag Features ──
        for lag in FIRE_LAGS:
            p[f'hotspot_lag_{lag}h'] = p['hotspot_count'].shift(lag)
            p[f'frp_sum_lag_{lag}h'] = p['frp_sum'].shift(lag)

        # ── PM2.5 Rolling Features ──
        for w in WINDOWS:
            p[f'pm25_roll_mean_{w}h'] = p['PM25'].shift(1).rolling(window=w, min_periods=1).mean()
            p[f'pm25_roll_std_{w}h']  = p['PM25'].shift(1).rolling(window=w, min_periods=1).std()
            p[f'pm25_roll_max_{w}h']  = p['PM25'].shift(1).rolling(window=w, min_periods=1).max()

        # ── Fire Rolling Features ──
        for w in [24, 48, 168]:
            p[f'hotspot_roll_sum_{w}h'] = p['hotspot_count'].shift(1).rolling(window=w, min_periods=1).sum()
            p[f'frp_roll_sum_{w}h']     = p['frp_sum'].shift(1).rolling(window=w, min_periods=1).sum()

        # ── Log Transforms ──
        p['hotspot_log']       = np.log1p(p['hotspot_count'])
        p['frp_sum_log']       = np.log1p(p['frp_sum'])
        p['frp_mean_log']      = np.log1p(p['frp_mean'])
        p['precipitation_log'] = np.log1p(p['precipitation'])

        for lag in FIRE_LAGS:
            p[f'hotspot_log_lag_{lag}h'] = np.log1p(p[f'hotspot_lag_{lag}h']).fillna(0)

        # ── Delta Features ──
        p['pm25_delta_1h']      = p['PM25'].diff(1).shift(1)
        p['pm25_delta_24h']     = p['PM25'].diff(24).shift(1)
        p['humidity_delta_1h']  = p['relative_humidity_2m'].diff(1)
        p['humidity_delta_24h'] = p['relative_humidity_2m'].diff(24)

        # ── Interaction Features ──
        p['temp_x_humidity'] = p['temperature_2m'] * p['relative_humidity_2m'] / 100
        p['hotspot_x_haze']  = p['hotspot_log'] * p['is_haze_season']
        p['frp_x_haze']      = p['frp_sum_log'] * p['is_haze_season']
        p['wind_x_hotspot']  = p['wind_speed_10m'] * p['hotspot_log']

        # ── Province Encoding ──
        p['province_label']      = PROVINCE_LABELS.get(prov, -1)
        p['province_target_enc'] = PROVINCE_MEAN_MAP.get(prov, 0)

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
    results = latest[["Province", "Datetime"]].copy()
    results.rename(columns={'Datetime': 'predict_datetime'}, inplace=True)

    for horizon in [1, 3, 6, 12, 24, 48, 72]:
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


def save_dashboard_data(df: pd.DataFrame, model, scaler, feature_list: list):
    """
    สร้าง pre-computed file ให้ Streamlit อ่านโดยตรง
    - ทุกแถวมี predicted แล้ว (ไม่ต้องโหลด model ใน dashboard)
    - เก็บแค่ 10 วันล่าสุด (rolling_168h ต้องการ 7 วัน + buffer)
    - บันทึกที่ data/processed/dashboard_data.csv
    """
    valid = df.dropna(subset=['pm25_lag_72h']).copy()

    for f in feature_list:
        if f not in valid.columns:
            valid[f] = 0

    X = valid[feature_list].fillna(0)
    X_scaled = scaler.transform(X)
    valid['predicted'] = model.predict(X_scaled)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=10)
    dashboard = valid[valid['Datetime'] >= cutoff].copy()

    base_cols = [
        'Datetime', 'Province', 'PM25', 'predicted',
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
        'hotspot_count', 'frp_sum', 'frp_mean',
    ]
    feat_cols = [f for f in feature_list if f not in base_cols]
    save_cols = [c for c in base_cols + feat_cols if c in dashboard.columns]

    out_path = DATA_DIR / 'processed' / 'dashboard_data.csv'
    dashboard[save_cols].to_csv(out_path, index=False)
    print(f"  Saved → dashboard_data.csv  ({len(dashboard):,} rows × {len(save_cols)} cols)")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running predict pipeline...")

    print("\n1. Loading artifacts...")
    model, scaler, feature_list = load_artifacts()

    print("\n2. Loading data...")
    df = load_data()

    print("\n3. Building features...")
    df = build_features(df)
    print(f"  Feature rows: {len(df):,}, columns: {df.shape[1]}")

    print("\n4. Predicting...")
    results = run_predict(df, model, scaler, feature_list)

    print("\n5. Saving predictions...")
    save_predictions(results)

    print("\n6. Saving dashboard data...")
    save_dashboard_data(df, model, scaler, feature_list)

    print("\nResult preview:")
    print(results[["Province", "pred_24h", "pred_48h", "pred_72h"]].to_string(index=False))
    print("\nDone.")
