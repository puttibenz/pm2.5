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

def fetch_openmeteo_weather(city_name, lat, lon, start_date, end_date):
    # สังเกตว่า URL จะเปลี่ยนเป็น archive-api สำหรับดึงข้อมูลอดีต
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # กำหนด Features ที่ต้องการดึงตามแผนของคุณเบนซ์ (+ ความเร็วลม)
    hourly_features = ",".join([
        "temperature_2m", 
        "relative_humidity_2m", 
        "precipitation", 
        "surface_pressure", 
        "wind_speed_10m", 
        "wind_direction_10m"
    ])
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_features,
        "timezone": "auto" # ปรับเวลาเป็น Timezone ท้องถิ่นอัตโนมัติ
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        data = response.json()
        
        # แปลงข้อมูล JSON เป็น Pandas DataFrame
        df = pd.DataFrame(data['hourly'])
        
        # เปลี่ยนชื่อคอลัมน์เวลา และแปลงเป็น Datetime Object ให้ตรงกับไฟล์ PM2.5
        df.rename(columns={'time': 'Datetime'}, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Province'] = city_name
        
        # จัดเรียงคอลัมน์ให้ดูง่ายขึ้น
        cols = ['Datetime', 'Province'] + [c for c in df.columns if c not in ['Datetime', 'Province']]
        df = df[cols]
        
        return df

    except Exception as e:
        print(f"❌ ดึงข้อมูลสภาพอากาศ {city_name} ไม่สำเร็จ: {e}")
        return None

if __name__ == "__main__":
    # แนะนำให้ดึงช่วงเวลาเดียวกันกับไฟล์ PM2.5 ที่เราดึงมาก่อนหน้านี้ จะได้นำมา Merge กันได้พอดีครับ
    start_date = '2023-01-01'
    end_date = '2025-12-31'
    
    print(f"🚀 เริ่มดึงข้อมูลสภาพอากาศ (Weather) ตั้งแต่ {start_date} ถึง {end_date}")
    
    for city, coords in NORTHERN_CITIES.items():
        print(f"📍 กำลังดึงข้อมูลสภาพอากาศจังหวัด: {city} ...", end=" ")
        
        df_weather = fetch_openmeteo_weather(city, coords['lat'], coords['lon'], start_date, end_date)
        
        if df_weather is not None and not df_weather.empty:
            save_path = RAW_DATA_DIR / f"openmeteo_weather_{city.replace(' ', '')}_{start_date[:4]}.csv"
            df_weather.to_csv(save_path, index=False)
            print(f"✅ สำเร็จ! (ได้ข้อมูลมา {len(df_weather)} บรรทัด)")
        
        # พัก 2 วินาที
        time.sleep(2)
        
    print("\n🎉 ดึงข้อมูลสภาพอากาศครบ 8 จังหวัดภาคเหนือเรียบร้อยแล้ว!")