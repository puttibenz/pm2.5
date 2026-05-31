# สิ่งที่ต้องแก้ไขในโปรเจกต์ PM2.5 Early Warning System

> เอกสารนี้ระบุ bug และจุดที่ต้องปรับปรุงทั้งหมด พร้อมโค้ดก่อน/หลังสำหรับแต่ละรายการ
>
> **หมายเหตุ:** โปรเจกต์นี้ใช้ XGBoost ซึ่งเป็น tree-based model — **ไม่จำเป็นต้อง scale ข้อมูลก่อน predict** เพราะ XGBoost ไม่ได้คำนวณ distance หรือ gradient จากค่าตัวเลขโดยตรง การ scale จึงไม่มีผลต่อ accuracy ลบ `scaler` ออกจาก codebase ได้เลย

---

## 🔴 Bug ระดับ Critical (ต้องแก้ก่อน)

---

### BUG-01: `scaler` โหลดมาแต่ไม่จำเป็นและทำให้ code สับสน

**ไฟล์:** `app/predict.py`, `app/main.py`

**ปัญหา:**
XGBoost ไม่ต้องการ feature scaling เพราะเป็น tree-based model ที่ตัดสินใจจาก threshold ไม่ใช่ distance/gradient แต่ปัจจุบัน code โหลด `scaler.pkl` มาเก็บไว้แล้วไม่ได้ใช้ทำให้คนอ่านโค้ดสับสนว่าควร scale หรือเปล่า

**โค้ดปัจจุบัน (สับสน):**
```python
# app/predict.py
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / 'xgboost_pm25.pkl')
    scaler = joblib.load(ARTIFACT_DIR / 'scaler.pkl')   # ❌ โหลดมาแต่ไม่ใช้
    feature_list = json.load(open(ARTIFACT_DIR / 'feature_list.json'))
    return model, scaler, feature_list

# main block
model, scaler, feature_list = load_artifacts()   # ❌ รับ scaler มาแต่ไม่ได้ใช้ต่อ
```

**วิธีแก้ — ลบ scaler ออกทั้งหมด:**
```python
# app/predict.py
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / 'xgboost_pm25.pkl')
    feature_list = json.load(open(ARTIFACT_DIR / 'feature_list.json'))
    return model, feature_list   # ✅ ไม่มี scaler

# main block
model, feature_list = load_artifacts()   # ✅ clean
```

ทำเหมือนกันใน `app/main.py` และลบไฟล์ `artifacts/scaler.pkl` ออกจาก repo ด้วย

---

### BUG-02: `is_haze_season` นิยามไม่ตรงกันระหว่าง inference และ display

**ไฟล์:** `app/predict.py` vs `app/components.py`

**ปัญหา:**
Feature `is_haze_season` ถูกสร้างต่างกันใน 2 ที่ ทำให้ค่าที่โมเดลใช้ predict กับค่าที่ Dashboard แสดงผลไม่ตรงกัน (Train-Serving Skew) ส่งผลให้ช่วงเดือนมกราคม–กุมภาพันธ์ prediction ผิดพลาดโดยไม่รู้ตัว

```python
# app/components.py — บรรทัด 106
d["is_haze_season"] = dt.dt.month.isin([1, 2, 3, 4]).astype(int)  # เดือน 1, 2, 3, 4
```

```python
# app/predict.py — บรรทัด 123
p.at[p.index[idx], 'is_haze_season'] = 1 if dt.month in [3, 4] else 0  # ❌ แค่เดือน 3, 4
```

**ขั้นตอนที่ 1 — ตรวจสอบก่อนว่าตอน train ใช้นิยามไหน** โดยดูใน `notebooks/modeling.ipynb` หรือ `src/preprocessing/` ว่า `is_haze_season` ถูกสร้างอย่างไร แล้วเลือกนิยามนั้นเป็นมาตรฐาน

**ขั้นตอนที่ 2 — สร้าง `app/config.py` เพื่อเก็บค่า constants ทั้งหมด:**

```python
# app/config.py  (ไฟล์ใหม่)

# ปรับตรงนี้ที่เดียว — ต้องตรงกับที่ใช้ตอน train
HAZE_MONTHS = [1, 2, 3, 4]

PROVINCE_COORDS = {
    "Chiang Mai":   {"lat": 18.7883, "lon": 98.9853},
    "Chiang Rai":   {"lat": 19.9105, "lon": 99.8253},
    "Mae Hong Son": {"lat": 19.3003, "lon": 97.9654},
    "Lamphun":      {"lat": 18.5745, "lon": 99.0087},
    "Lampang":      {"lat": 18.2888, "lon": 99.4930},
    "Phayao":       {"lat": 19.1666, "lon": 99.9022},
    "Phrae":        {"lat": 18.1446, "lon": 100.1403},
    "Nan":          {"lat": 18.7756, "lon": 100.7730},
}

PROVINCES = list(PROVINCE_COORDS.keys())
```

**ขั้นตอนที่ 3 — แก้ทั้งสองไฟล์ให้ import จาก config:**

```python
# app/predict.py — แก้บรรทัด 123
from app.config import HAZE_MONTHS
...
p.at[p.index[idx], 'is_haze_season'] = 1 if dt.month in HAZE_MONTHS else 0  # ✅
```

```python
# app/components.py — แก้บรรทัด 106
from app.config import HAZE_MONTHS
...
d["is_haze_season"] = dt.dt.month.isin(HAZE_MONTHS).astype(int)  # ✅
```

---

### BUG-03: Code ซ้ำใน `predict.py` — `run_recursive_predict()` define แต่ไม่ถูกเรียกใช้

**ไฟล์:** `app/predict.py`

**ปัญหา:**
มีฟังก์ชัน `run_recursive_predict()` ถูก define ไว้ในบรรทัด 187–218 แต่ใน `if __name__ == "__main__":` (บรรทัด 252+) กลับรัน loop เหมือนกันทุกอย่างซ้ำอีกรอบโดยไม่เรียกฟังก์ชันนั้น ทำให้มี logic เดียวกันอยู่ 2 ที่ — ถ้าแก้ที่หนึ่งอาจลืมแก้อีกที่

นอกจากนี้ฟังก์ชัน `run_recursive_predict()` return แค่ `results` (DataFrame เดียว) แต่ main block ต้องการทั้ง `results` และ `df_final` เพื่อส่งเข้า `save_predictions()` ทำให้ไม่สามารถเรียกฟังก์ชันได้จริง

**โค้ดปัจจุบัน (ผิด):**
```python
# ฟังก์ชันที่ define ไว้ — return แค่ results
def run_recursive_predict(df, model, feature_list):
    results_list = []
    for prov in PROVINCES:
        p = df[df['Province'] == prov].copy()...
        for i in range(last_actual_idx + 1, len(p)):
            X = build_features_single_row(p, i, feature_list)
            pred = model.predict(X)[0]
            p.at[i, 'PM25'] = pred
            results_list.append({...})
        # ❌ ไม่ได้เก็บ p กลับมา (df_final หาย)
    return pd.DataFrame(results_list)   # return แค่อันเดียว


if __name__ == "__main__":
    model, scaler, feature_list = load_artifacts()
    df = load_data()
    results_list = []
    df_final_list = []
    # ❌ รัน loop ซ้ำทั้งหมดแทนที่จะเรียก run_recursive_predict()
    for prov in PROVINCES:
        p = df[df['Province'] == prov].copy()...
        for i in range(last_actual_idx + 1, len(p)):
            X = build_features_single_row(p, i, feature_list)
            pred = model.predict(X)[0]
            ...
```

**วิธีแก้ — แก้ฟังก์ชันให้ return ทั้ง results และ df_final แล้วเรียกจาก main:**
```python
def run_recursive_predict(df, model, feature_list):
    """ทำนายทีละชั่วโมงแบบ Recursive สำหรับทุกจังหวัด"""
    results_list = []
    df_final_list = []   # ✅ เพิ่ม list เก็บ df แต่ละจังหวัด

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
        df_final_list.append(p)   # ✅ เก็บ p กลับมา

    return pd.DataFrame(results_list), pd.concat(df_final_list, ignore_index=True)  # ✅ return 2 ค่า


if __name__ == "__main__":
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running recursive predict pipeline...")

    model, feature_list = load_artifacts()   # ✅ ไม่มี scaler (จาก BUG-01)
    df = load_data()

    print("\nStarting Recursive Prediction (7 Days)...")
    results, df_final = run_recursive_predict(df, model, feature_list)  # ✅ เรียกฟังก์ชัน

    save_predictions(results, df_final)

    print("\nPreview Prediction (Chiang Mai):")
    print(results[results['Province'] == 'Chiang Mai'].head(10))
    print("\nDone.")
```

---

## 🟠 ปัญหาระดับกลาง (ควรแก้)

---

### IMPROVE-01: `PROVINCE_COORDS` / `NORTHERN_CITIES` กระจายอยู่ 4 ที่

**ไฟล์ที่มีพิกัดซ้ำ:**
- `app/fetch_daily.py` — ชื่อ `NORTHERN_CITIES`
- `app/components.py` — ชื่อ `PROVINCE_COORDS`
- `src/data_collection/fetch_open_meteo.py` — ชื่อ `NORTHERN_CITIES`
- `src/data_collection/fetch_forecast.py` — ชื่อ `NORTHERN_CITIES`

**ปัญหา:** ถ้าต้องการเพิ่มจังหวัดใหม่หรือแก้พิกัด ต้องแก้ 4 ไฟล์ และเสี่ยงต่อการไม่สอดคล้องกัน

**วิธีแก้:** ใช้ `app/config.py` ที่สร้างจาก BUG-02 แล้ว import ทุกที่:

```python
# ลบการ define NORTHERN_CITIES / PROVINCE_COORDS ในทุกไฟล์
# แล้วแทนด้วย:
from app.config import PROVINCE_COORDS, PROVINCES
```

ถ้า `app/config.py` import ไม่ได้จาก `src/` ให้ย้ายไปเป็น `config.py` ที่ root แทน

---

### IMPROVE-02: `requirements.txt` ไม่ระบุ version

**ไฟล์:** `requirements.txt`

**ปัญหา:** ไม่มีการ pin version ทำให้ `pip install` อาจดึง version ใหม่ที่ไม่ compatible มาติดตั้ง และ GitHub Actions อาจ build พังได้โดยไม่มีสาเหตุชัดเจน

**โค้ดปัจจุบัน:**
```
pandas
numpy
xgboost
scikit-learn
streamlit
...
```

**วิธีแก้:** รัน `pip freeze` ใน virtual environment ที่ทำงานได้อยู่ แล้วเอา version มาใส่:

```bash
pip freeze > requirements.txt
```

ผลลัพธ์ที่ได้ควรมีหน้าตาแบบนี้:
```
pandas==2.2.2
numpy==1.26.4
requests==2.31.0
python-dotenv==1.0.1
xgboost==2.0.3
scikit-learn==1.4.2
streamlit==1.35.0
plotly==5.22.0
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
geopandas==0.14.4
shapely==2.0.4
shap==0.45.1
tqdm==4.66.4
```

ให้ลบ `jupyter` และ `selenium` ออกจาก `requirements.txt` เพราะเป็น dev dependency ทำให้ GitHub Actions ติดตั้งช้าโดยไม่จำเป็น แยกไว้ใน `requirements-dev.txt` แทน:

```
# requirements-dev.txt
jupyter
selenium
```

---

### IMPROVE-03: `save_predictions()` — คอลัมน์ `predicted` ไม่แยกระหว่างค่าจริงและค่าพยากรณ์

**ไฟล์:** `app/predict.py` บรรทัด 237

**ปัญหา:**
```python
df_final['predicted'] = df_final['PM25']   # ❌ copy ค่า PM25 ทุก row ทั้งค่าจริงและค่าพยากรณ์
```
ผลคือ Dashboard ไม่สามารถแยกได้ว่าแถวไหนเป็นค่า sensor จริงและแถวไหนเป็น prediction ทำให้ graph แสดงผลไม่ถูกต้อง

**วิธีแก้ — เพิ่ม flag `is_predicted` ใน `run_recursive_predict()`:**
```python
def run_recursive_predict(df, model, feature_list):
    ...
    for prov in PROVINCES:
        p = df[df['Province'] == prov].copy()...
        p['is_predicted'] = False   # ✅ default = ค่าจริงทุก row

        for i in range(last_actual_idx + 1, len(p)):
            X = build_features_single_row(p, i, feature_list)
            pred = model.predict(X)[0]
            p.at[i, 'PM25'] = pred
            p.at[i, 'is_predicted'] = True   # ✅ mark แถวที่ predict
            results_list.append({...})
        df_final_list.append(p)
```

จากนั้นใน `save_predictions()` ลบบรรทัดนี้ออก:
```python
# ❌ ลบบรรทัดนี้ออก
df_final['predicted'] = df_final['PM25']
```

Dashboard ใช้ `is_predicted` column แทนได้เลย:
```python
# ใน components.py — แยกกราฟค่าจริงกับค่าพยากรณ์
actual    = df[df['is_predicted'] == False]
predicted = df[df['is_predicted'] == True]
```

---

### IMPROVE-04: ไม่มีการ log Model Metrics ใน Pipeline

**ไฟล์:** `.github/workflows/daily_fetch.yml`

**ปัญหา:**
Pipeline รันและบันทึก prediction ทุกวัน แต่ไม่มีการบันทึกว่า model ทำงานได้ดีแค่ไหน ไม่รู้ว่า MAE/RMSE เป็นเท่าไหร่ และไม่รู้เมื่อ model เริ่ม drift หรือ accuracy แย่ลง

**วิธีแก้ — สร้าง `app/evaluate_daily.py` ใหม่:**

```python
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

    hist = pd.read_csv(history_path, parse_dates=['Datetime'])
    pred_yesterday = hist[hist['Datetime'].dt.date == yesterday]

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
        on=['Province', 'Datetime'],
        suffixes=('_pred', '_actual')
    )

    if merged.empty:
        print("No matching rows to evaluate.")
        return

    mae  = mean_absolute_error(merged['PM25_actual'], merged['PM25_pred'])
    rmse = mean_squared_error(merged['PM25_actual'], merged['PM25_pred'], squared=False)

    print(f"\n📊 Model Performance ({yesterday}):")
    print(f"   MAE  = {mae:.2f} µg/m³")
    print(f"   RMSE = {rmse:.2f} µg/m³")
    print(f"   N    = {len(merged)} samples")

    # บันทึก metrics ลง CSV เพื่อ track ตลอดเวลา
    row = pd.DataFrame([{
        'date':      yesterday,
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
    print(f"   Saved → {METRICS_PATH}")


if __name__ == "__main__":
    evaluate_yesterday()
```

เพิ่ม step ใน `daily_fetch.yml`:
```yaml
- name: Evaluate yesterday predictions
  run: python app/evaluate_daily.py

- name: Commit metrics
  run: |
    git add predictions/model_metrics.csv
    git diff --cached --quiet || git commit -m "metrics: update daily evaluation $(date +%Y-%m-%d)"
```

---

## สรุปลำดับการแก้ไขที่แนะนำ

| ลำดับ | รายการ | ไฟล์ | ความสำคัญ |
|---|---|---|---|
| 1 | BUG-01: ลบ `scaler` ออกจาก `load_artifacts()` และ main block | `app/predict.py`, `app/main.py` | 🔴 Critical |
| 2 | BUG-02: ตรวจสอบ `is_haze_season` ว่า train ใช้เดือนอะไร แล้วรวมไว้ใน `config.py` | `app/config.py` (ใหม่), `app/predict.py`, `app/components.py` | 🔴 Critical |
| 3 | BUG-03: แก้ `run_recursive_predict()` ให้ return 2 ค่า และเรียกจาก main block | `app/predict.py` | 🔴 Critical |
| 4 | IMPROVE-03: เพิ่ม `is_predicted` flag และลบ `df_final['predicted'] = df_final['PM25']` | `app/predict.py` | 🟠 High |
| 5 | IMPROVE-01: ลบ `PROVINCE_COORDS` / `NORTHERN_CITIES` ที่ซ้ำกัน ให้ import จาก `config.py` | `app/fetch_daily.py`, `app/components.py`, `src/data_collection/*` | 🟠 High |
| 6 | IMPROVE-02: Pin version ใน `requirements.txt` และแยก dev deps ออก | `requirements.txt`, `requirements-dev.txt` (ใหม่) | 🟠 High |
| 7 | IMPROVE-04: สร้าง `evaluate_daily.py` และเพิ่มใน GitHub Actions | `app/evaluate_daily.py` (ใหม่), `.github/workflows/daily_fetch.yml` | 🟡 Medium |
