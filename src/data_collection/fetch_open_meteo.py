from pathlib import Path

import pandas as pd
import requests
import time

# พิกัดคร่าวๆ ของใจกลางเมือง 8 จังหวัดภาคเหนือ
NORTHERN_CITIES = {
    "Chiang Mai": {"lat": 18.7883, "lon": 98.9853},
    "Chiang Rai": {"lat": 19.9105, "lon": 99.8253},
    "Mae Hong Son": {"lat": 19.3003, "lon": 97.9654},
    "Lamphun": {"lat": 18.5745, "lon": 99.0087},
    "Lampang": {"lat": 18.2888, "lon": 99.4930},
    "Phayao": {"lat": 19.1666, "lon": 99.9022},
    "Phrae": {"lat": 18.1446, "lon": 100.1403},
    "Nan": {"lat": 18.7756, "lon": 100.7730}
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def fetch_openmeteo_pm25(city_name, lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5",
        "timezone": "auto" # จัดการ timezone ให้อัตโนมัติ (เป็นเวลาไทย)
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status() # ดักจับ Error หากยิงไม่ผ่าน
        data = response.json()
        
        # Open-Meteo คืนค่ามาเป็น Dictionary จัดการยัดลง Pandas ได้ง่ายมาก
        df = pd.DataFrame()
        df['Datetime'] = pd.to_datetime(data['hourly']['time'])
        df['PM25'] = data['hourly']['pm2_5']
        df['Province'] = city_name
        
        return df

    except Exception as e:
        print(f"❌ ดึงข้อมูล {city_name} ไม่สำเร็จ: {e}")
        return None

if __name__ == "__main__":
    # ดึงข้อมูลปี 2023 (สามารถเปลี่ยนเป็นปี 2019-2023 เพื่อให้ได้ Data เยอะขึ้นตอนเทรนโมเดลได้เลย)
    start_date = '2023-01-01'
    end_date = '2025-12-31'
    
    print(f"🚀 เริ่มดึงข้อมูล PM2.5 จาก Open-Meteo ตั้งแต่ {start_date} ถึง {end_date}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for city, coords in NORTHERN_CITIES.items():
        print(f"📍 กำลังดึงข้อมูลจังหวัด: {city} ...", end=" ")
        
        df_city = fetch_openmeteo_pm25(city, coords['lat'], coords['lon'], start_date, end_date)
        
        if df_city is not None and not df_city.empty:
            save_path = RAW_DATA_DIR / f"openmeteo_pm25_{city.replace(' ', '')}_{start_date[:4]}.csv"
            df_city.to_csv(save_path, index=False)
            print(f"✅ สำเร็จ! (ได้ข้อมูลมา {len(df_city)} บรรทัด)")
        else:
            print("⚠️ ไม่พบข้อมูลหรือดึงข้อมูลไม่สำเร็จ")
        
        # พัก 2 วินาทีก่อนยิงจังหวัดต่อไป ป้องกันเซิร์ฟเวอร์เตะ
        time.sleep(2)
        
    print("\n🎉 ดึงข้อมูลครบ 8 จังหวัดภาคเหนือเรียบร้อยแล้ว!")