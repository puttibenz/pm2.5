# Pipeline: Load Data → Recursive Feature Engineering → Load Model → Predict 7 Days → Save
# ═══════════════════════════════════════════════════════════════

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

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

# ── Feature constants ─────────────────────────────────────────
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
FIRE_LAGS = [24, 48, 72]
WINDOWS   = [3, 6, 12, 24, 48, 168]

PROVINCE_LABELS = {
    'Chiang Mai': 0, 'Chiang Rai': 1, 'Lampang': 2, 'Lamphun': 3,
    'Mae Hong Son': 4, 'Nan': 5, 'Phayao': 6, 'Phrae': 7,
}

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

def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / 'xgboost_pm25.pkl')
    scaler = joblib.load(ARTIFACT_DIR / 'scaler.pkl')
    with open(ARTIFACT_DIR / 'feature_list.json', encoding='utf-8') as f:
        feature_list = json.load(f)
    return model, scaler, feature_list

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
    สร้าง Features สำหรับแถวเดียว (Recursive)
    """
    # ดึงข้อมูลมาเฉพาะส่วนที่จำเป็นเพื่อความเร็ว
    # WINDOWS สูงสุดคือ 168 ดังนั้นต้องย้อนหลังอย่างน้อย 168
    p = df_prov.iloc[max(0, current_idx-170):current_idx+1].copy()
    idx = len(p) - 1
    
    # Time Features
    dt = p.iloc[idx]['Datetime']
    p.at[p.index[idx], 'year']           = dt.year
    p.at[p.index[idx], 'month']          = dt.month
    p.at[p.index[idx], 'day']            = dt.day
    p.at[p.index[idx], 'hour']           = dt.hour
    p.at[p.index[idx], 'dayofyear']      = dt.dayofyear
    p.at[p.index[idx], 'is_haze_season'] = 1 if dt.month in [3, 4] else 0
    p.at[p.index[idx], 'hour_sin']       = np.sin(2 * np.pi * dt.hour  / 24)
    p.at[p.index[idx], 'hour_cos']       = np.cos(2 * np.pi * dt.hour  / 24)
    p.at[p.index[idx], 'month_sin']      = np.sin(2 * np.pi * dt.month / 12)
    p.at[p.index[idx], 'month_cos']      = np.cos(2 * np.pi * dt.month / 12)
    p.at[p.index[idx], 'day_sin']        = np.sin(2 * np.pi * dt.day   / 365)
    p.at[p.index[idx], 'day_cos']        = np.cos(2 * np.pi * dt.day   / 365)
    
    wd = p.iloc[idx]['wind_direction_10m']
    p.at[p.index[idx], 'wind_dir_sin']   = np.sin(np.radians(wd))
    p.at[p.index[idx], 'wind_dir_cos']   = np.cos(np.radians(wd))

    # PM2.5 Lag
    for lag in LAG_HOURS:
        if idx >= lag:
            p.at[p.index[idx], f'pm25_lag_{lag}h'] = p.iloc[idx-lag]['PM25']
            
    # Fire Lag
    for lag in FIRE_LAGS:
        if idx >= lag:
            p.at[p.index[idx], f'hotspot_lag_{lag}h'] = p.iloc[idx-lag]['hotspot_count']
            p.at[p.index[idx], f'frp_sum_lag_{lag}h'] = p.iloc[idx-lag]['frp_sum']

    # PM2.5 Rolling
    for w in WINDOWS:
        if idx >= 1:
            window_data = p['PM25'].iloc[max(0, idx-w):idx]
            p.at[p.index[idx], f'pm25_roll_mean_{w}h'] = window_data.mean()
            p.at[p.index[idx], f'pm25_roll_std_{w}h']  = window_data.std()
            p.at[p.index[idx], f'pm25_roll_max_{w}h']  = window_data.max()

    # Fire Rolling
    for w in [24, 48, 168]:
        if idx >= 1:
            window_data = p['hotspot_count'].iloc[max(0, idx-w):idx]
            p.at[p.index[idx], f'hotspot_roll_sum_{w}h'] = window_data.sum()
            window_frp = p['frp_sum'].iloc[max(0, idx-w):idx]
            p.at[p.index[idx], f'frp_roll_sum_{w}h']     = window_frp.sum()

    # Log/Interaction
    p.at[p.index[idx], 'hotspot_log']       = np.log1p(p.iloc[idx]['hotspot_count'])
    p.at[p.index[idx], 'frp_sum_log']       = np.log1p(p.iloc[idx]['frp_sum'])
    p.at[p.index[idx], 'frp_mean_log']      = np.log1p(p.iloc[idx]['frp_mean'])
    p.at[p.index[idx], 'precipitation_log'] = np.log1p(p.iloc[idx]['precipitation'])
    
    for lag in FIRE_LAGS:
        val = p.at[p.index[idx], f'hotspot_lag_{lag}h'] if f'hotspot_lag_{lag}h' in p.columns else 0
        p.at[p.index[idx], f'hotspot_log_lag_{lag}h'] = np.log1p(val)

    p.at[p.index[idx], 'pm25_delta_1h']      = p.iloc[idx-1]['PM25'] - p.iloc[idx-2]['PM25'] if idx >= 2 else 0
    p.at[p.index[idx], 'pm25_delta_24h']     = p.iloc[idx-1]['PM25'] - p.iloc[idx-25]['PM25'] if idx >= 25 else 0
    p.at[p.index[idx], 'humidity_delta_1h']  = p.iloc[idx]['relative_humidity_2m'] - p.iloc[idx-1]['relative_humidity_2m'] if idx >= 1 else 0
    p.at[p.index[idx], 'humidity_delta_24h'] = p.iloc[idx]['relative_humidity_2m'] - p.iloc[idx-24]['relative_humidity_2m'] if idx >= 24 else 0
    
    p.at[p.index[idx], 'temp_x_humidity'] = p.iloc[idx]['temperature_2m'] * p.iloc[idx]['relative_humidity_2m'] / 100
    p.at[p.index[idx], 'hotspot_x_haze']  = p.at[p.index[idx], 'hotspot_log'] * p.at[p.index[idx], 'is_haze_season']
    p.at[p.index[idx], 'frp_x_haze']      = p.at[p.index[idx], 'frp_sum_log'] * p.at[p.index[idx], 'is_haze_season']
    p.at[p.index[idx], 'wind_x_hotspot']  = p.iloc[idx]['wind_speed_10m'] * p.at[p.index[idx], 'hotspot_log']

    p.at[p.index[idx], 'province_label']      = PROVINCE_LABELS.get(p.iloc[idx]['Province'], -1)
    p.at[p.index[idx], 'province_target_enc'] = PROVINCE_MEAN_MAP.get(p.iloc[idx]['Province'], 0)
    
    return p.iloc[idx][feature_list].fillna(0).to_frame().T.astype(float)

def run_recursive_predict(df, model, feature_list):
    """
    ทำนายทีละชั่วโมงแบบ Recursive
    """
    results_list = []
    
    for prov in PROVINCES:
        print(f"  Predicting for {prov}...")
        p = df[df['Province'] == prov].copy().sort_values('Datetime').reset_index(drop=True)
        
        # หาจุดเริ่มต้นของ Forecast (แถวที่ PM25 เป็น NaN)
        # หรือถ้ามีค่า PM25 ล่าสุดเมื่อไหร่ ให้เริ่มจากตรงนั้น
        last_actual_idx = p[p['PM25'].notna()].index.max()
        
        # เราจะทำนายตั้งแต่วินาทีถัดไปจากค่าจริงล่าสุด
        for i in range(last_actual_idx + 1, len(p)):
            # 1. สร้าง Features สำหรับแถว i
            X = build_features_single_row(p, i, feature_list)
            
            # 2. Predict
            pred = model.predict(X)[0]
            
            # 3. ใส่ผลลัพธ์กลับลงไปใน PM25 เพื่อใช้ใน Loop ถัดไป
            p.at[i, 'PM25'] = pred
            
            # เก็บผลลัพธ์เฉพาะส่วนที่เป็นพยากรณ์
            results_list.append({
                'Province': prov,
                'Datetime': p.at[i, 'Datetime'],
                'Predicted_PM25': pred
            })
            
    return pd.DataFrame(results_list)

def save_predictions(results, df_final):
    # บันทึกเป็น CSV สำหรับแสดงผลประวัติพยากรณ์
    out_path = OUTPUT_DIR / "predictions_7d.csv"
    results.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    # บันทึกไฟล์สำหรับ Dashboard (รวมประวัติย้อนหลัง + พยากรณ์)
    dashboard_path = DATA_DIR / "processed" / "dashboard_data.csv"
    
    # เลือกคอลัมน์ที่จำเป็นสำหรับ Dashboard
    base_cols = [
        'Datetime', 'Province', 'PM25', 'temperature_2m', 'relative_humidity_2m', 
        'precipitation', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
        'hotspot_count', 'frp_sum', 'frp_mean',
    ]
    # รวม PM25 ที่เป็นทั้งค่าจริงและค่าพยากรณ์ไว้ในคอลัมน์ PM25 และเพิ่มคอลัมน์ 'predicted'
    df_final['predicted'] = df_final['PM25'] 
    
    # กรองเอาแค่ 14 วัน (ย้อนหลัง 7 + พยากรณ์ 7)
    cutoff = datetime.now() - timedelta(days=14)
    dashboard_df = df_final[df_final['Datetime'] >= cutoff].copy()
    
    # เก็บ Features บางตัวที่ Dashboard ใช้แสดงผล (เช่น SHAP)
    # ในที่นี้เก็บไว้ท้ังหมดที่อยู่ใน feature_list
    save_cols = base_cols + ['predicted'] + [f for f in df_final.columns if f in PROVINCE_LABELS or 'lag' in f or 'roll' in f or 'delta' in f or 'log' in f]
    save_cols = list(set([c for c in save_cols if c in df_final.columns]))
    
    dashboard_df[save_cols].to_csv(dashboard_path, index=False)
    print(f"  Saved → {dashboard_path} ({len(dashboard_df)} rows)")

if __name__ == "__main__":
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running recursive predict pipeline...")
    
    model, scaler, feature_list = load_artifacts()
    df = load_data()
    
    print("\nStarting Recursive Prediction (7 Days)...")
    # เราต้องการ Dataframe p ที่อัปเดตค่า PM25 แล้วกลับมาด้วย
    results_list = []
    df_final_list = []
    
    for prov in PROVINCES:
        print(f"  Predicting for {prov}...")
        p = df[df['Province'] == prov].copy().sort_values('Datetime').reset_index(drop=True)
        last_actual_idx = p[p['PM25'].notna()].index.max()
        
        for i in range(last_actual_idx + 1, len(p)):
            X = build_features_single_row(p, i, feature_list)
            pred = model.predict(X)[0]
            p.at[i, 'PM25'] = pred
            results_list.append({
                'Province': prov,
                'Datetime': p.at[i, 'Datetime'],
                'Predicted_PM25': pred
            })
        df_final_list.append(p)
            
    results = pd.DataFrame(results_list)
    df_final = pd.concat(df_final_list, ignore_index=True)
    
    save_predictions(results, df_final)
    
    # Preview
    print("\nPreview Prediction (Chiang Mai):")
    print(results[results['Province'] == 'Chiang Mai'].head(10))
    print("\nDone.")
