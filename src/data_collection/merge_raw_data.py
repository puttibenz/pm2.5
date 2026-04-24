from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def merge_all_raw_data():
    # อ่านไฟล์ PM2.5 ทั้ง 8 จังหวัด แล้วรวมกัน
    pm25_files = sorted(RAW_DATA_DIR.glob("openmeteo_pm25_*.csv"))
    df_pm25 = pd.concat([pd.read_csv(f) for f in pm25_files], ignore_index=True)
    print(f"📊 PM2.5: {len(df_pm25)} แถว จาก {len(pm25_files)} ไฟล์")

    # อ่านไฟล์ Weather ทั้ง 8 จังหวัด แล้วรวมกัน
    weather_files = sorted(RAW_DATA_DIR.glob("openmeteo_weather_*.csv"))
    df_weather = pd.concat([pd.read_csv(f) for f in weather_files], ignore_index=True)
    print(f"🌤️  Weather: {len(df_weather)} แถว จาก {len(weather_files)} ไฟล์")

    # Merge PM2.5 + Weather บน Datetime และ Province
    df_merged = pd.merge(df_pm25, df_weather, on=["Datetime", "Province"], how="inner")
    print(f"✅ Merged: {len(df_merged)} แถว, {len(df_merged.columns)} คอลัมน์")
    print(f"   คอลัมน์: {list(df_merged.columns)}")

    # บันทึกเป็นไฟล์เดียว
    save_path = RAW_DATA_DIR / "openmeteo_all_provinces_2023-.csv"
    df_merged.to_csv(save_path, index=False)
    print(f"💾 บันทึกเรียบร้อยที่: {save_path}")

if __name__ == "__main__":
    merge_all_raw_data()
