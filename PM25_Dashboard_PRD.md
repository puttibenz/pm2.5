# PRD: PM2.5 Early Warning System — Dashboard Improvements
**Version:** 1.0  
**Date:** 2026-05-05  
**Status:** Draft  
**Owner:** PM2.5 Early Warning Team  

---

## 1. Context & Background

ระบบ PM2.5 Early Warning System สำหรับภาคเหนือของไทย (8 จังหวัด) ปัจจุบันประกอบด้วย:

- **Data Pipeline:** fetch_daily.py → predict.py → dashboard_data.csv
- **Model:** XGBoost พยากรณ์ล่วงหน้า 7 วัน แบบ recursive (69 features)
- **Dashboard:** Streamlit app แสดง forecast, alert, hotspot map, SHAP explainability
- **Automation:** GitHub Actions รันทุกวัน 07:00 ICT

PRD นี้ระบุการปรับปรุงที่ควรทำ แบ่งเป็น 3 ระดับ: **Critical Bugs**, **UX/UI Improvements**, **New Features**

---

## 2. Problem Statement

### 2.1 Critical Bugs (ต้องแก้ก่อน deploy)

| ID | ปัญหา | ผลกระทบ |
|----|--------|---------|
| BUG-01 | `df_final['predicted'] = df_final['PM25']` ใน predict.py ทำให้ค่า predicted = actual ทุก row | กราฟพยากรณ์แสดงค่าผิด ผู้ใช้ไม่ได้เห็น forecast จริง |
| BUG-02 | `is_haze_season` นิยามต่างกัน: components.py ใช้ [1,2,3,4] แต่ predict.py ใช้ [3,4] | Feature drift ระหว่าง training และ inference |
| BUG-03 | `save_cols` filter ด้วย `f in PROVINCE_LABELS` (province name dict) ทำให้ feature columns หายออกจาก dashboard_data.csv | Tab 4 SHAP ใช้ X ที่ fill=0 แทน feature จริง |
| BUG-04 | Timezone mismatch: plot_7day_forecast คำนวณ `now` ภายใน function เอง แต่ server อาจอยู่ UTC | เส้น "ปัจจุบัน" อาจคลาดเคลื่อน 7 ชั่วโมง |
| BUG-05 | fetch_daily.py ดึง forecast แค่ 3 วัน แต่ recursive predict ต้องการ 7 วัน วันที่ 4–7 ไม่มีข้อมูลอากาศจริง | Prediction accuracy ลดลงช่วง day 4–7 |

### 2.2 UX Problems

- ผู้ใช้ต้องเลือกจังหวัดทีละจังหวัดเพื่อเช็ค ไม่มี overview 8 จังหวัดในหน้าเดียว
- ไม่มีระบบแจ้งเตือน (push/notify) เมื่อค่าเกิน threshold
- SHAP คำนวณใหม่ทุกครั้งที่เปิด Tab ทำให้รอ 3–5 วินาที
- ปฏิทิน 7 วัน ใช้ `st.columns(7)` ล้นบนหน้าจอแคบ
- Sidebar แสดงเวลาอัปเดต hardcode ไม่ใช่จาก data จริง

---

## 3. Goals & Non-Goals

### Goals
- แก้ bugs ทั้งหมดใน Section 2.1 ให้ค่า predicted ถูกต้อง
- เพิ่ม Province Overview เพื่อให้เห็นภาพรวมทั้ง 8 จังหวัดพร้อมกัน
- ลด SHAP compute time จาก ~5s เป็น <0.5s ด้วย precompute
- เพิ่ม Line Notify เมื่อ PM2.5 คาดการณ์เกิน threshold
- ขยาย FIRMS บounding box ให้ครอบ cross-border hotspot (พม่า/ลาว)

### Non-Goals
- ไม่เปลี่ยน model architecture ใน scope นี้
- ไม่ refactor ไปใช้ FastAPI หรือ framework อื่น
- ไม่เพิ่ม real-time streaming (ยังเป็น daily batch)

---

## 4. Requirements

### 4.1 BUG FIXES

#### BUG-01: แก้ predicted column
```python
# ก่อน save dashboard_data, merge ค่า Predicted_PM25 กลับเข้า df_final
results_map = results.set_index(['Province', 'Datetime'])['Predicted_PM25']
df_final['predicted'] = df_final.apply(
    lambda r: results_map.get((r['Province'], r['Datetime']), r['PM25']),
    axis=1
)
```

**Acceptance Criteria:**
- แถวที่ Datetime > last_actual_idx มีค่า `predicted` ≠ `PM25`
- กราฟ Tab 1 แสดงเส้น forecast แยกออกจากเส้น actual ชัดเจน

#### BUG-02: Sync is_haze_season
```python
# ใช้ definition เดียวกันทั้ง pipeline: Jan–Apr = haze season
IS_HAZE_MONTHS = [1, 2, 3, 4]
is_haze_season = int(dt.month in IS_HAZE_MONTHS)
```

**Acceptance Criteria:**
- ทั้ง `components.py` และ `predict.py` import/ใช้ constant เดียวกัน
- เพิ่ม unit test ตรวจค่า Jan=1, May=0

#### BUG-03: แก้ save_cols
```python
DASHBOARD_COLS = [
    'Datetime', 'Province', 'PM25', 'predicted',
    'temperature_2m', 'relative_humidity_2m', 'precipitation',
    'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
    'hotspot_count', 'frp_sum', 'frp_mean',
] + [f for f in feature_list if any(k in f for k in ['lag', 'roll', 'delta', 'log', 'sin', 'cos', 'enc'])]
```

**Acceptance Criteria:**
- dashboard_data.csv มี feature columns ครบตาม feature_list
- Tab 4 SHAP แสดงค่า feature จริง ไม่ใช่ 0

#### BUG-04: Timezone fix
```python
# ส่ง now เป็น parameter แทนให้ function คำนวณเอง
now = pd.Timestamp.now(tz='Asia/Bangkok').tz_localize(None).floor('h')
plot_7day_forecast(prov_data, province, now=now)
```

#### BUG-05: Extend forecast fetch
```python
# fetch_daily.py: ดึง Open-Meteo 7 วันล่วงหน้า (ไม่ใช่แค่ 3)
end = (TODAY + timedelta(days=7)).isoformat()
```

---

### 4.2 FEATURE: Province Overview Dashboard

**User Story:**  
ในฐานะเจ้าหน้าที่สาธารณสุข ฉันต้องการเห็นภาพรวมฝุ่น PM2.5 ของทั้ง 8 จังหวัดในหน้าเดียว เพื่อตัดสินใจว่าจังหวัดใดต้องการการแก้ไขเร่งด่วน

**Requirements:**
- แสดง ranking card 8 จังหวัด เรียงตาม max predicted PM2.5 ในช่วง 24 ชั่วโมงข้างหน้า
- แต่ละ card แสดง: ชื่อจังหวัด, ค่า PM2.5 ปัจจุบัน, ค่าพยากรณ์สูงสุด 24h, color badge ตามระดับ
- วางไว้เหนือ province selector ใน sidebar หรือเป็น tab แรกแยก
- คลิก card แล้ว province selector เปลี่ยนตาม

**Data Source:** `all_data` ที่โหลดอยู่แล้ว group by Province

**Acceptance Criteria:**
- โหลดเร็ว ไม่ทำ API call เพิ่ม (ใช้ data ที่ cache ไว้)
- Color coding ตรงกับ LEVELS ใน components.py
- Responsive บน mobile (ไม่ overflow)

---

### 4.3 FEATURE: Line Notify Alert

**User Story:**  
ในฐานะเจ้าหน้าที่ ฉันต้องการรับ LINE notification อัตโนมัติเมื่อ PM2.5 คาดการณ์ 24 ชั่วโมงข้างหน้าเกินมาตรฐาน เพื่อเตรียมรับมือล่วงหน้า

**Requirements:**
- เพิ่ม step ใน GitHub Actions หลัง `predict.py` รันเสร็จ
- ส่ง LINE Notify เมื่อมีจังหวัดใดที่ max_predicted_24h > 50 µg/m³
- Message format:
  ```
  🚨 แจ้งเตือน PM2.5 ภาคเหนือ [วันที่]
  
  จังหวัดที่น่าเป็นห่วง (24h ข้างหน้า):
  🔴 เชียงราย: 87.3 µg/m³ (มีผลต่อสุขภาพ)
  🟠 ลำปาง: 52.1 µg/m³ (เริ่มมีผลต่อสุขภาพ)
  
  ดูรายละเอียด: [Dashboard URL]
  ```
- เก็บ `LINE_NOTIFY_TOKEN` ใน GitHub Secrets
- ถ้า token ไม่มีให้ skip ไม่ให้ pipeline fail

**Thresholds:**
| ระดับ | PM2.5 | Action |
|-------|-------|--------|
| ปานกลาง | > 37.5 | แจ้งเตือนกลุ่มเสี่ยง |
| เริ่มมีผล | > 50 | แจ้งเตือนทั่วไป |
| อันตราย | > 75 | แจ้งเตือนเร่งด่วน |

**Implementation:** สร้างไฟล์ `app/notify.py` แยก เพื่อให้ test ได้อิสระ

---

### 4.4 FEATURE: SHAP Precompute

**User Story:**  
ในฐานะผู้ใช้ dashboard ฉันต้องการให้ Tab SHAP โหลดทันทีโดยไม่ต้องรอ เพื่อ workflow ที่ลื่นไหล

**Requirements:**
- เพิ่ม `compute_shap_values()` ใน `predict.py` รันหลัง recursive predict เสร็จ
- Save SHAP values เป็น `data/processed/shap_latest.csv` (Province × Feature × SHAP value)
- Dashboard อ่านจากไฟล์แทนคำนวณ live
- Fallback: ถ้าไฟล์ไม่มีให้ compute live แบบเดิม (แต่แสดง spinner warning)

**Schema ของ shap_latest.csv:**
```
Province, feature_name, shap_value, feature_value, base_value, predicted_pm25
```

**Acceptance Criteria:**
- Tab 4 โหลดภายใน 0.5 วินาที
- ค่า SHAP ตรงกับ `model.predict(X_latest)` ±0.1 µg/m³

---

### 4.5 FEATURE: Cross-border Hotspot

**User Story:**  
ในฐานะนักวิเคราะห์ ฉันต้องการเห็น hotspot จากพม่าและลาวในแผนที่ เนื่องจากไฟในประเทศเพื่อนบ้านส่งผลต่อ PM2.5 ในเชียงรายและแม่ฮ่องสอนโดยตรง

**Requirements:**
- ขยาย FIRMS bounding box จาก `97.3,17.5,102.5,20.5` เป็น `92.0,13.0,106.0,26.0` (ครอบ GMS)
- เพิ่มคอลัมน์ `country` ใน firms_recent_hotspots.csv โดย spatial join กับ country shapefile
- แผนที่ใน Tab 3 แสดง hotspot ต่างประเทศด้วยสัญลักษณ์แตกต่าง (รูปร่าง/ขอบ)
- Filter ใน sidebar: เลือก "ไทยเท่านั้น" / "ทั้ง GMS"
- Priority score ยังคำนวณเหมือนเดิม (distance จาก province centroid)

**Data:**
- Country shapefile: ใช้ Natural Earth 1:10m Admin 0 (free, license OK)
- ไม่ต้อง aggregate by country สำหรับ predict.py (ยังใช้แค่ภาคเหนือไทย)

---

### 4.6 FEATURE: Model Accuracy Tracker

**User Story:**  
ในฐานะ data scientist ฉันต้องการตรวจสอบว่าโมเดลยัง accurate แค่ไหนเมื่อเวลาผ่านไป เพื่อตัดสินใจว่าควร retrain หรือยัง

**Requirements:**
- เพิ่ม Tab ใหม่ "📈 Model Performance" ใน dashboard
- แสดง MAE, RMSE รายสัปดาห์ ย้อนหลัง 8 สัปดาห์
- เปรียบเทียบ predictions_history.csv กับ actual PM25 ใน openmeteo_all_provinces.csv
- แสดง error distribution histogram per province
- แจ้งเตือนถ้า MAE สัปดาห์ล่าสุด > 15 µg/m³ (signal to retrain)

**Data Available:** `predictions/predictions_history.csv` มีอยู่แล้ว สามารถ join กับ actual ได้

---

### 4.7 UX IMPROVEMENTS

#### 4.7.1 Sidebar: Last Updated Timestamp
```python
# แสดงวันที่จริงจาก data
last_update = all_data['Datetime'].max()
st.caption(f"อัปเดตล่าสุด: {last_update.strftime('%d %b %Y %H:%M')} ICT")
```

#### 4.7.2 Mobile-Responsive Calendar
- เปลี่ยน `st.columns(len(daily_max))` เป็น grid ที่ scroll ได้
- จำกัดสูงสุด 4 columns บนหน้าต่างแคบ

#### 4.7.3 Error Boundaries
- ครอบ data loading ทุก section ด้วย try/except
- แสดง `st.warning()` แทน crash เมื่อ file ไม่พร้อม

#### 4.7.4 KPI Card Consistency
- ทุก card ที่เกี่ยวกับ PM2.5 ใช้ `delta_color="inverse"` (สูง = แดง)
- เพิ่ม tooltip อธิบาย unit และ threshold

---

## 5. Technical Architecture

### 5.1 File Changes Summary

| ไฟล์ | การเปลี่ยนแปลง |
|------|----------------|
| `app/predict.py` | แก้ BUG-01, BUG-03; เพิ่ม SHAP precompute |
| `app/components.py` | แก้ BUG-02; เพิ่ม `IS_HAZE_MONTHS` constant; เพิ่ม province ranking component |
| `app/main.py` | แก้ BUG-04; เพิ่ม Tab Province Overview; เพิ่ม Tab Model Performance; แก้ UX 4.7.x |
| `app/fetch_daily.py` | แก้ BUG-05 (extend forecast 7d); ขยาย FIRMS bbox |
| `app/notify.py` | ไฟล์ใหม่: LINE Notify logic |
| `.github/workflows/daily_fetch.yml` | เพิ่ม step notify หลัง predict |
| `src/data_collection/fetch_forecast.py` | ตรวจสอบว่าดึงครบ 7 วัน |

### 5.2 New Files

```
data/processed/
  shap_latest.csv          # SHAP values precomputed รายจังหวัด
  
app/
  notify.py                # LINE Notify integration
  
notebooks/gadm_world/
  ne_10m_admin_0.shp       # Natural Earth country boundaries (GMS)
```

### 5.3 Constants ที่ควร Extract เป็น config

```python
# config.py (ใหม่)
IS_HAZE_MONTHS = [1, 2, 3, 4]
PM25_THRESHOLDS = [25, 37.5, 50, 75]
FIRMS_BBOX_THAILAND = "97.3,17.5,102.5,20.5"
FIRMS_BBOX_GMS = "92.0,13.0,106.0,26.0"
FORECAST_DAYS = 7
NOTIFY_THRESHOLD_MODERATE = 37.5
NOTIFY_THRESHOLD_HIGH = 50
NOTIFY_THRESHOLD_DANGER = 75
```

---

## 6. Implementation Plan

### Phase 1: Bug Fixes (Priority: Critical, ทำก่อน)
- [ ] BUG-01: แก้ predicted column merge logic
- [ ] BUG-02: Sync IS_HAZE_MONTHS
- [ ] BUG-03: แก้ save_cols filter
- [ ] BUG-04: Timezone fix ใน plot_7day_forecast
- [ ] BUG-05: Extend forecast fetch เป็น 7 วัน
- [ ] Test pipeline end-to-end หลังแก้ bugs

### Phase 2: Performance & UX (2–3 สัปดาห์)
- [ ] SHAP precompute ใน predict.py
- [ ] Dashboard อ่าน SHAP จากไฟล์
- [ ] Province Ranking component
- [ ] UX improvements (4.7.1–4.7.4)

### Phase 3: New Features (1–2 เดือน)
- [ ] LINE Notify integration
- [ ] Cross-border hotspot (GMS bbox + country label)
- [ ] Model Accuracy Tracker tab

---

## 7. Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| SHAP Tab load time | ~5 วินาที | < 0.5 วินาที |
| Bug BUG-01: predicted ≠ actual | ไม่ผ่าน | ผ่าน unit test |
| Province overview load time | ไม่มี | < 1 วินาที |
| MAE ของ predictions (เทียบ actual) | ไม่ tracking | tracking รายสัปดาห์ |
| Alert delivery (LINE) เมื่อ PM2.5 > 50 | ไม่มี | ส่งภายใน 5 นาทีหลัง predict.py รัน |

---

## 8. Open Questions

1. **Air4Thai Integration:** PM2.5 "actual" ปัจจุบันมาจาก Open-Meteo (model) ไม่ใช่ sensor จริง — ควร integrate Air4Thai API ใน Phase ถัดไปหรือไม่?
2. **Retrain Schedule:** เมื่อ MAE เกิน threshold ควร trigger retrain อัตโนมัติหรือ manual?
3. **LINE Notify vs LINE OA:** ถ้าต้องการ segment ผู้รับตามจังหวัดควรเปลี่ยนเป็น LINE OA + Rich Menu
4. **GMS Shapefile License:** ตรวจสอบ Natural Earth license ให้ชัดก่อนใช้เชิงพาณิชย์

---

## 9. References

- โปรเจกต์ปัจจุบัน: `pm25-early-warning-cnx/`
- Open-Meteo API docs: https://open-meteo.com/en/docs
- NASA FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/
- LINE Notify API: https://notify-bot.line.me/doc/en/
- Air4Thai API: http://air4thai.pcd.go.th/webV3/
- Natural Earth Data: https://www.naturalearthdata.com/

---

*PRD นี้เขียนโดยอ้างอิงจาก code review ของ codebase เวอร์ชัน 2026-05-05*
