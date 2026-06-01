# Pipeline: Load Data → Recursive Feature Engineering → Load Model → Predict 7 Days → Save
# ═══════════════════════════════════════════════════════════════

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from config import IS_HAZE_MONTHS, PROVINCES, PROVINCE_LABELS, PROVINCE_MEAN_MAP

# ── Config ────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / 'artifacts'
DATA_DIR     = REPO_ROOT / 'data'
OUTPUT_DIR   = REPO_ROOT / 'predictions'
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Feature constants ─────────────────────────────────────────
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FIRE_LAGS = [24, 48, 72]
WINDOWS   = [3, 6, 12, 24, 48, 168]

def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / 'xgboost_pm25.pkl')
    # Scaler is not used as model was trained on raw data
    with open(ARTIFACT_DIR / 'feature_list.json', encoding='utf-8') as f:
        feature_list = json.load(f)
    return model, feature_list

def load_data():
    """
    โหลดข้อมูลอดีต + ข้อมูลพยากรณ์อากาศ 7 วัน
    """
    # 1. โหลดข้อมูลอดีต (Meteo + PM2.5)
    meteo_path = DATA_DIR / "raw" / "openmeteo_all_provinces.csv"
    if not meteo_path.exists():
        raise FileNotFoundError("ไม่พบไฟล์ openmeteo_all_provinces.csv")
    
    hist_df = pd.read_csv(meteo_path, parse_dates=['Datetime'])
    # เอาแค่ 10 วันล่าสุดเพื่อทำ Lag/Rolling
    cutoff = pd.Timestamp.now(tz='Asia/Bangkok').replace(tzinfo=None) - pd.Timedelta(days=10)
    hist_df = hist_df[hist_df['Datetime'] >= cutoff].copy()

    # เคลียร์ค่า PM2.5 ในอนาคต (พยากรณ์ล่วงหน้าจาก API) เพื่อให้โมเดลทำนายแบบ Recursive ตั้งแต่เวลาปัจจุบันเป็นต้นไป
    now_th = pd.Timestamp.now(tz='Asia/Bangkok').replace(tzinfo=None).floor('h')
    hist_df.loc[hist_df['Datetime'] >= now_th, 'PM25'] = np.nan

    # 2. โหลดข้อมูลพยากรณ์อากาศ (ที่สร้างใหม่)
    forecast_path = DATA_DIR / "raw" / "openmeteo_forecast_7d.csv"
    if forecast_path.exists():
        forecast_df = pd.read_csv(forecast_path, parse_dates=['Datetime'])
        print(f"  Loaded forecast: {len(forecast_df)} rows")
    else:
        print("  WARN: ไม่พบ openmeteo_forecast_7d.csv — จะใช้ข้อมูลจาก fetch_daily (3 วัน) แทน")
        forecast_df = hist_df[hist_df['Datetime'] > hist_df['Datetime'].max() - pd.Timedelta(hours=1)].copy() # Placeholder

    # 3. โหลด Hotspot
    hotspot_path = DATA_DIR / "processed" / "firms_daily_by_province.csv"
    if hotspot_path.exists():
        hotspot = pd.read_csv(hotspot_path, parse_dates=['date'])
    else:
        hotspot = pd.DataFrame(columns=['date', 'Province', 'hotspot_count', 'frp_sum', 'frp_mean'])

    # รวมข้อมูล
    # สำหรับ Forecast เราจะสมมติ Hotspot = ค่าเฉลี่ย 3 วันล่าสุด
    recent_hotspot = hotspot[hotspot['date'] >= hotspot['date'].max() - pd.Timedelta(days=3)]
    hotspot_proxy = recent_hotspot.groupby('Province')[['hotspot_count', 'frp_sum', 'frp_mean']].mean().reset_index()

    # เตรียม Dataframe หลัก
    # รวม Historical + Forecast
    full_meteo = pd.concat([hist_df, forecast_df], ignore_index=True).drop_duplicates(['Datetime', 'Province'])
    full_meteo = full_meteo.sort_values(['Province', 'Datetime']).reset_index(drop=True)
    
    full_meteo['date'] = full_meteo['Datetime'].dt.normalize()
    df = full_meteo.merge(hotspot, on=['date', 'Province'], how='left')
    
    # เติม Hotspot อนาคตด้วย Proxy
    for prov in PROVINCES:
        mask = (df['Province'] == prov) & (df['hotspot_count'].isna())
        prov_proxy = hotspot_proxy[hotspot_proxy['Province'] == prov]
        if not prov_proxy.empty:
            df.loc[mask, 'hotspot_count'] = prov_proxy['hotspot_count'].values[0]
            df.loc[mask, 'frp_sum'] = prov_proxy['frp_sum'].values[0]
            df.loc[mask, 'frp_mean'] = prov_proxy['frp_mean'].values[0]
            
    df[['hotspot_count', 'frp_sum', 'frp_mean']] = df[['hotspot_count', 'frp_sum', 'frp_mean']].fillna(0)
    return df

def build_features_single_row(df_prov, current_idx, feature_list):
    """
    สร้าง Features สำหรับแถวเดียว (Recursive) - เวอร์ชันปรับปรุงความเร็ว
    """
    # ดึงข้อมูลเฉพาะส่วนที่จำเป็น
    start = max(0, current_idx - 170)
    window = df_prov.iloc[start:current_idx + 1]
    idx = len(window) - 1
    row = window.iloc[idx]
    dt = row['Datetime']

    features = {}
    
    # Base features from row
    base_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
        'hotspot_count', 'frp_sum', 'frp_mean'
    ]
    for col in base_cols:
        if col in row:
            features[col] = row[col]

    # Time features
    features['year']           = dt.year
    features['month']          = dt.month
    features['day']            = dt.day
    features['hour']           = dt.hour
    features['dayofyear']      = dt.dayofyear
    features['is_haze_season'] = 1 if dt.month in IS_HAZE_MONTHS else 0
    features['hour_sin']       = np.sin(2 * np.pi * dt.hour / 24)
    features['hour_cos']       = np.cos(2 * np.pi * dt.hour / 24)
    features['month_sin']      = np.sin(2 * np.pi * dt.month / 12)
    features['month_cos']      = np.cos(2 * np.pi * dt.month / 12)
    features['day_sin']        = np.sin(2 * np.pi * dt.day / 31)
    features['day_cos']        = np.cos(2 * np.pi * dt.day / 31)

    # Wind
    wd = row['wind_direction_10m']
    features['wind_dir_sin'] = np.sin(np.radians(wd))
    features['wind_dir_cos'] = np.cos(np.radians(wd))

    # PM2.5 Lag
    pm25_series = window['PM25'].values
    for lag in LAG_HOURS:
        features[f'pm25_lag_{lag}h'] = pm25_series[idx - lag] if idx >= lag else 0

    # Fire Lag
    hotspot_series = window['hotspot_count'].values
    frp_series = window['frp_sum'].values
    for lag in FIRE_LAGS:
        if idx >= lag:
            features[f'hotspot_lag_{lag}h'] = hotspot_series[idx - lag]
            features[f'frp_sum_lag_{lag}h'] = frp_series[idx - lag]
            features[f'hotspot_log_lag_{lag}h'] = np.log1p(hotspot_series[idx - lag])
        else:
            features[f'hotspot_lag_{lag}h'] = 0
            features[f'frp_sum_lag_{lag}h'] = 0
            features[f'hotspot_log_lag_{lag}h'] = 0

    # PM2.5 Rolling
    for w in WINDOWS:
        slice_ = pm25_series[max(0, idx - w):idx]
        if len(slice_) > 0:
            features[f'pm25_roll_mean_{w}h'] = slice_.mean()
            features[f'pm25_roll_std_{w}h']  = slice_.std() if len(slice_) > 1 else 0
            features[f'pm25_roll_max_{w}h']  = slice_.max()
        else:
            features[f'pm25_roll_mean_{w}h'] = 0
            features[f'pm25_roll_std_{w}h']  = 0
            features[f'pm25_roll_max_{w}h']  = 0

    # Fire Rolling
    for w in [24, 48, 168]:
        slice_h = hotspot_series[max(0, idx - w):idx]
        slice_f = frp_series[max(0, idx - w):idx]
        features[f'hotspot_roll_sum_{w}h'] = slice_h.sum() if len(slice_h) > 0 else 0
        features[f'frp_roll_sum_{w}h']     = slice_f.sum() if len(slice_f) > 0 else 0

    # Log/Interaction
    features['hotspot_log']       = np.log1p(row['hotspot_count'])
    features['frp_sum_log']       = np.log1p(row['frp_sum'])
    features['frp_mean_log']      = np.log1p(row['frp_mean'])
    features['precipitation_log'] = np.log1p(row['precipitation'])
    
    features['pm25_delta_1h']      = pm25_series[idx-1] - pm25_series[idx-2] if idx >= 2 else 0
    features['pm25_delta_24h']     = pm25_series[idx-1] - pm25_series[idx-25] if idx >= 25 else 0
    features['humidity_delta_1h']  = row['relative_humidity_2m'] - window.iloc[idx-1]['relative_humidity_2m'] if idx >= 1 else 0
    features['humidity_delta_24h'] = row['relative_humidity_2m'] - window.iloc[idx-24]['relative_humidity_2m'] if idx >= 24 else 0
    
    features['temp_x_humidity'] = row['temperature_2m'] * row['relative_humidity_2m'] / 100
    features['hotspot_x_haze']  = features['hotspot_log'] * features['is_haze_season']
    features['frp_x_haze']      = features['frp_sum_log'] * features['is_haze_season']
    features['wind_x_hotspot']  = row['wind_speed_10m'] * features['hotspot_log']

    features['province_label']      = PROVINCE_LABELS.get(row['Province'], -1)
    features['province_target_enc'] = PROVINCE_MEAN_MAP.get(row['Province'], 0)
    
    # แปลงเป็น DataFrame และเรียงคอลัมน์ตาม feature_list
    X = pd.DataFrame([features])
    return X[feature_list].fillna(0).astype(float)

def run_recursive_predict(df, model, feature_list):
    """
    ทำนายทีละชั่วโมงแบบ Recursive สำหรับทุกจังหวัด
    """
    results_list = []
    df_final_list = []
    
    now = pd.Timestamp.now(tz='Asia/Bangkok').tz_localize(None).floor('h')

    for prov in PROVINCES:
        print(f"  Predicting for {prov}...")
        p = df[df['Province'] == prov].copy().sort_values('Datetime').reset_index(drop=True)
        p['is_predicted'] = False
        p['predicted'] = p['PM25']

        # หาจุดเริ่มต้นของ Forecast 
        actual_indices = p[p['PM25'].notna()].index
        if len(actual_indices) == 0:
            last_actual_idx = -1
        else:
            last_actual_idx = int(actual_indices.max())
        
        for i in range(last_actual_idx + 1, len(p)):
            # 1. สร้าง Features
            X = build_features_single_row(p, i, feature_list)
            
            # 2. Predict (Raw data, no scaler)
            pred = model.predict(X)[0]
            pred = max(0.0, float(pred))  # ✓ ป้องกันค่าติดลบ
            
            # 3. อัปเดตค่าเพื่อใช้ใน Loop ถัดไป
            p.at[i, 'PM25'] = pred
            p.at[i, 'predicted'] = pred
            p.at[i, 'is_predicted'] = True
            
            # อัปเดต Features อื่นๆ ลงใน p สำหรับ SHAP และ Loop ถัดไป (ถ้าจำเป็น)
            for col in X.columns:
                p.at[i, col] = X.iloc[0][col]
            
            results_list.append({
                'Province': prov,
                'Datetime': p.at[i, 'Datetime'],
                'Predicted_PM25': pred
            })
        df_final_list.append(p)
            
    return pd.DataFrame(results_list), pd.concat(df_final_list, ignore_index=True)

def save_predictions(results, df_final, feature_list):
    # บันทึกเป็น CSV สำหรับแสดงผลประวัติพยากรณ์
    out_path = OUTPUT_DIR / "predictions_7d.csv"
    results.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    # บันทึกไฟล์สำหรับ Dashboard (รวมประวัติย้อนหลัง + พยากรณ์)
    dashboard_path = DATA_DIR / "processed" / "dashboard_data.csv"
    
    # แยก Predicted ออกจาก PM25 จริง
    results_map = results.set_index(['Province', 'Datetime'])['Predicted_PM25']
    df_final['predicted'] = df_final.apply(
        lambda r: results_map.get((r['Province'], r['Datetime']), r['PM25']),
        axis=1
    )
    
    # สำหรับแถวที่เป็นพยากรณ์ ให้ PM25 เป็น NaN (เพื่อให้กราฟ Actual ตัดจบที่ปัจจุบัน)
    forecast_mask = df_final['is_predicted'] == True
    df_final.loc[forecast_mask, 'PM25'] = np.nan

    # กรองคอลัมน์ที่จำเป็นสำหรับ Dashboard และ SHAP
    base_cols = [
        'Datetime', 'Province', 'PM25', 'predicted', 'is_predicted',
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
        'hotspot_count', 'frp_sum', 'frp_mean',
    ]
    # เก็บ Features ที่ SHAP ใช้ โดยเช็คจาก feature_list
    shap_features = [f for f in feature_list if any(k in f for k in ['lag', 'roll', 'delta', 'log', 'sin', 'cos', 'enc', 'haze', 'year', 'month', 'day', 'hour'])]
    save_cols = list(set(base_cols + shap_features))
    save_cols = [c for c in save_cols if c in df_final.columns]
    
    # กรองเอาแค่ 14 วัน (ย้อนหลัง 7 + พยากรณ์ 7)
    max_dt = df_final['Datetime'].max()
    cutoff = max_dt - timedelta(days=14)
    dashboard_df = df_final[df_final['Datetime'] >= cutoff].copy()
    
    dashboard_df[save_cols].to_csv(dashboard_path, index=False)
    print(f"  Saved → {dashboard_path} ({len(dashboard_df)} rows)")

def compute_shap_values(model, df_final, feature_list):
    """
    Precompute SHAP values for the latest prediction row of each province.
    Saves to data/processed/shap_latest.csv
    """
    try:
        import shap
    except ImportError:
        print("  WARN: Library 'shap' not found. Skipping precompute.")
        return

    print("\nPrecomputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_records = []

    for prov in PROVINCES:
        p = df_final[df_final['Province'] == prov].sort_values('Datetime').reset_index(drop=True)
        if p.empty: continue
        
        # Build features for the latest row using build_features_single_row
        X = build_features_single_row(p, len(p) - 1, feature_list)
        
        # คำนวณ SHAP
        shap_values = explainer.shap_values(X)
        base_value  = float(explainer.expected_value)
        pred_val    = float(p['predicted'].iloc[-1])
        
        sv = shap_values[0]
        for i, feat in enumerate(feature_list):
            shap_records.append({
                'Province': prov,
                'feature_name': feat,
                'shap_value': sv[i],
                'feature_value': X.iloc[0, i],
                'base_value': base_value,
                'predicted_pm25': pred_val
            })

    shap_df = pd.DataFrame(shap_records)
    out_path = DATA_DIR / "processed" / "shap_latest.csv"
    shap_df.to_csv(out_path, index=False)
    print(f"  Saved SHAP → {out_path}")

if __name__ == "__main__":
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running recursive predict pipeline...")
    
    model, feature_list = load_artifacts()
    df = load_data()
    
    print("\nStarting Recursive Prediction (7 Days)...")
    results, df_final = run_recursive_predict(df, model, feature_list)
    
    save_predictions(results, df_final, feature_list)
    compute_shap_values(model, df_final, feature_list)
    
    # Preview
    print("\nPreview Prediction (Chiang Mai):")
    print(results[results['Province'] == 'Chiang Mai'].head(10))
    print("\nDone.")
