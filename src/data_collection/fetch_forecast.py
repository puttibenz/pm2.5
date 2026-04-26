import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

# พิกัด 8 จังหวัดภาคเหนือ (อ้างอิงตามโครงสร้างเดิมของโปรเจกต์)
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

def fetch_weather_forecast(city_name, lat, lon):
    """
    ดึงข้อมูลพยากรณ์อากาศล่วงหน้า 7 วันจาก Open-Meteo Weather Forecast API
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
        "forecast_days": 7
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # แปลงข้อมูล JSON เป็น DataFrame
        df = pd.DataFrame()
        df['Datetime'] = pd.to_datetime(data['hourly']['time'])
        df['temperature_2m'] = data['hourly']['temperature_2m']
        df['relative_humidity_2m'] = data['hourly']['relative_humidity_2m']
        df['precipitation'] = data['hourly']['precipitation']
        df['surface_pressure'] = data['hourly']['surface_pressure']
        df['wind_speed_10m'] = data['hourly']['wind_speed_10m']
        df['wind_direction_10m'] = data['hourly']['wind_direction_10m']
        df['Province'] = city_name
        
        return df
    except Exception as e:
        print(f"❌ Error fetching forecast for {city_name}: {e}")
        return None

if __name__ == "__main__":
    print(f"🚀 เริ่มดึงข้อมูลพยากรณ์อากาศ 7 วันล่วงหน้า... ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_forecasts = []
    for city, coords in NORTHERN_CITIES.items():
        print(f"📍 ดึงข้อมูล: {city}...", end=" ", flush=True)
        df = fetch_weather_forecast(city, coords['lat'], coords['lon'])
        if df is not None:
            all_forecasts.append(df)
            print(f"✅ สำเร็จ ({len(df)} rows)")
        time.sleep(1) # พักเพื่อไม่ให้โดน Rate Limit
        
    if all_forecasts:
        full_df = pd.concat(all_forecasts, ignore_index=True)
        save_path = RAW_DATA_DIR / "openmeteo_forecast_7d.csv"
        full_df.to_csv(save_path, index=False)
        print(f"\n🎉 บันทึกข้อมูลพยากรณ์ทั้งหมดลงใน: {save_path}")
    else:
        print("\n⚠️ ไม่สามารถดึงข้อมูลพยากรณ์ได้เลย")
