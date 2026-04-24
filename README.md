# PM2.5 Early Warning System - Chiang Mai

ระบบเตือนภัยล่วงหน้าค่าฝุ่น PM2.5 สำหรับจังหวัดเชียงใหม่

## โครงสร้างโปรเจกต์

```
pm25-early-warning-cnx/
├── data/               # ข้อมูล (raw, processed, features)
├── notebooks/          # Jupyter Notebooks สำหรับ EDA
├── src/                # Source code หลัก (Data Pipeline)
├── app/                # Streamlit Dashboard
├── .env                # API Keys (ห้ามอัปโหลด)
├── .gitignore
├── requirements.txt
└── README.md
```

## วิธีติดตั้ง

1. Clone repository:
   ```bash
   git clone <repo-url>
   cd pm25-early-warning-cnx
   ```

2. สร้าง Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. ติดตั้ง Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. ตั้งค่า API Keys ในไฟล์ `.env`

## วิธีรัน

### รัน Data Pipeline
```bash
python src/data_collection/fetch_air4thai.py
python src/data_collection/fetch_nasa_firms.py
python src/data_collection/fetch_open_meteo.py
python src/preprocessing/build_features.py
```

### เทรนโมเดล
```bash
python src/modeling/train_xgboost.py
python src/modeling/evaluate.py
```

### รัน Dashboard
```bash
streamlit run app/main.py
```

## แหล่งข้อมูล

- **Air4Thai** - ข้อมูลคุณภาพอากาศจากกรมควบคุมมลพิษ
- **NASA FIRMS** - ข้อมูลจุดความร้อน (Hotspot)
- **Open-Meteo** - ข้อมูลสภาพอากาศ
