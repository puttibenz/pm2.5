# 🌫️ PM2.5 Early Warning System — ภาคเหนือไทย

ระบบเตือนภัยล่วงหน้าค่าฝุ่น PM2.5 สำหรับ 8 จังหวัดภาคเหนือ พยากรณ์ล่วงหน้า 7 วัน ด้วยโมเดล XGBoost และข้อมูลจาก NASA FIRMS + Open-Meteo API อัปเดตอัตโนมัติทุกวันผ่าน GitHub Actions

---

## ✨ ฟีเจอร์หลัก

- **พยากรณ์ PM2.5 ล่วงหน้า 7 วัน** ด้วย Recursive Forecasting (รายชั่วโมง)
- **แผนที่จุดเสี่ยง (Hotspot Priority Map)** คำนวณลำดับความสำคัญจาก FRP, ระยะทาง และทิศทางลม
- **ระบบแจ้งเตือน** แบ่งระดับตามมาตรฐาน PM2.5 ของไทย พร้อมปฏิทินความเสี่ยง 7 วัน
- **SHAP Explainability** อธิบายการตัดสินใจของโมเดลในแต่ละการพยากรณ์
- **Pipeline อัตโนมัติ** ดึงข้อมูลใหม่ + พยากรณ์ + บันทึกผลทุกวัน 07:00 ICT ผ่าน GitHub Actions

---

## 🗺️ จังหวัดที่รองรับ

เชียงใหม่ · เชียงราย · ลำปาง · ลำพูน · แม่ฮ่องสอน · น่าน · พะเยา · แพร่

---

## 📁 โครงสร้างโปรเจกต์

```
pm2.5/
├── .github/
│   └── workflows/
│       └── daily_fetch.yml     # GitHub Actions — รันทุกวัน 07:00 ICT
│
├── app/
│   ├── main.py                 # Streamlit Dashboard (หน้าหลัก)
│   ├── components.py           # UI components, charts, SHAP plots
│   ├── fetch_daily.py          # ดึงข้อมูล Open-Meteo + NASA FIRMS รายวัน
│   └── predict.py              # Recursive 7-day prediction pipeline
│
├── src/
│   ├── data_collection/
│   │   ├── fetch_open_meteo.py         # ดึงข้อมูลอุตุนิยมวิทยาย้อนหลัง
│   │   ├── fetch_forecast.py           # ดึงข้อมูลพยากรณ์อากาศ 7 วัน
│   │   ├── fetch_nasa_firms.py         # ดึงข้อมูล Hotspot จาก NASA FIRMS
│   │   └── merge_raw_data.py           # รวมข้อมูลจากหลายแหล่ง
│   ├── modeling/
│   │   ├── train_xgboost.py            # เทรนโมเดล XGBoost
│   │   └── evaluate.py                 # ประเมินผลโมเดล
│   └── preprocessing/                  # Feature engineering
│
├── artifacts/
│   ├── xgboost_pm25.pkl        # โมเดล XGBoost (trained)
│   ├── scaler.pkl              # StandardScaler
│   └── feature_list.json       # รายชื่อ features 69 ตัว
│
├── data/
│   ├── raw/                    # ข้อมูลดิบจาก API (ไม่ track ใน git)
│   └── processed/              # ข้อมูลที่ผ่านการประมวลผลแล้ว
│
├── predictions/
│   ├── predictions_7d.csv      # ผลพยากรณ์ 7 วันล่าสุด
│   ├── predictions_latest.csv  # ผลพยากรณ์ล่าสุด
│   └── predictions_history.csv # ประวัติการพยากรณ์
│
├── notebooks/
│   ├── cleaning.ipynb          # Data cleaning
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── merge.ipynb             # การรวมข้อมูล
│   └── modeling.ipynb          # การสร้างและ evaluate โมเดล
│
├── picture/                    # รูปภาพ EDA และ SHAP plots
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ วิธีติดตั้ง

### 1. Clone repository

```bash
git clone https://github.com/puttibenz/pm2.5.git
cd pm2.5
```

### 2. สร้าง Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 4. ตั้งค่า API Key

สร้างไฟล์ `.env` ในโฟลเดอร์หลัก:

```env
MAP_KEY=your_nasa_firms_api_key_here
```

> รับ API Key ได้ฟรีที่ [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key/)

---

## 🚀 วิธีรัน

### รันครั้งแรก (ดึงข้อมูลย้อนหลัง)

```bash
# 1. ดึงข้อมูลอุตุนิยมวิทยาย้อนหลัง
python src/data_collection/fetch_open_meteo.py

# 2. ดึงข้อมูล Hotspot ย้อนหลัง
python src/data_collection/fetch_nasa_firms.py

# 3. รวมข้อมูล
python src/data_collection/merge_raw_data.py
```

### เทรนโมเดล (ถ้าต้องการ retrain)

```bash
python src/modeling/train_xgboost.py
python src/modeling/evaluate.py
```

### รัน Daily Pipeline (เหมือนที่ GitHub Actions รันทุกวัน)

```bash
# ดึงข้อมูลวันนี้
python app/fetch_daily.py

# ดึงพยากรณ์อากาศ 7 วัน
python src/data_collection/fetch_forecast.py

# สร้างพยากรณ์ PM2.5
python app/predict.py
```

### เปิด Dashboard

```bash
streamlit run app/main.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501`

---

## 🤖 โมเดลและ Features

| รายละเอียด | ค่า |
|---|---|
| Algorithm | XGBoost Regressor |
| จำนวน Features | 69 |
| Forecasting Method | Recursive (hour-by-hour) |
| Horizon | 7 วัน (168 ชั่วโมง) |

### Feature Groups

- **PM2.5 Lag** — ค่าฝุ่นย้อนหลัง 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h
- **Rolling Statistics** — mean, std, max ใน windows 3h, 6h, 12h, 24h, 48h, 168h
- **Fire/Hotspot** — hotspot count, FRP sum/mean + lag + rolling จาก NASA FIRMS
- **Meteorology** — อุณหภูมิ, ความชื้น, ความกดอากาศ, ความเร็วลม, ทิศทางลม
- **Cyclical Encoding** — hour/month (sin+cos), wind direction (sin+cos)
- **Interaction Features** — hotspot × haze season, FRP × haze, wind × hotspot
- **Seasonal** — is_haze_season (ม.ค.–เม.ย.), delta features (1h, 24h)

---

## 🔄 GitHub Actions Pipeline

Pipeline รันอัตโนมัติทุกวัน **07:00 น. เวลาไทย (00:00 UTC)**

```
ดึงข้อมูล Open-Meteo + FIRMS  →  พยากรณ์ 7 วัน  →  commit CSV กลับ repo
```

กดรันมือได้จาก **Actions → Daily PM2.5 Data Fetch → Run workflow**

### ตั้งค่า Secret

ไปที่ `Settings → Secrets and variables → Actions` แล้วเพิ่ม:

| Secret | ค่า |
|---|---|
| `MAP_KEY` | NASA FIRMS API Key |

---

## 📡 แหล่งข้อมูล

| แหล่ง | ข้อมูล | ความถี่ |
|---|---|---|
| [NASA FIRMS VIIRS](https://firms.modaps.eosdis.nasa.gov/) | จุดความร้อน (Hotspot), Fire Radiative Power | รายวัน |
| [Open-Meteo](https://open-meteo.com/) | อุณหภูมิ, ความชื้น, ลม, ความกดอากาศ | รายชั่วโมง |
| [Open-Meteo Air Quality](https://open-meteo.com/) | PM2.5 (ค่าจริง + พยากรณ์) | รายชั่วโมง |

---

## 📊 ระดับ PM2.5 (มาตรฐานไทย)

| ระดับ | ค่า (µg/m³) | ความหมาย |
|---|---|---|
| 🟢 ดีมาก | 0 – 25 | ทำกิจกรรมกลางแจ้งได้ตามปกติ |
| 🟡 ดี | 26 – 37 | อากาศดี ไม่มีผลต่อสุขภาพ |
| 🟠 ปานกลาง | 38 – 50 | กลุ่มเสี่ยงควรลดกิจกรรมกลางแจ้ง |
| 🔴 เริ่มมีผลต่อสุขภาพ | 51 – 90 | ทุกคนควรลดกิจกรรมกลางแจ้ง |
| 🟣 มีผลต่อสุขภาพ | > 90 | หยุดทุกกิจกรรมกลางแจ้ง สวมหน้ากาก N95 |

---

## 🛠️ Tech Stack

- **ML** — XGBoost, scikit-learn, SHAP
- **Dashboard** — Streamlit, Plotly
- **Data** — pandas, numpy, geopandas, shapely
- **Automation** — GitHub Actions
- **APIs** — NASA FIRMS, Open-Meteo

---

## 📝 License

MIT License