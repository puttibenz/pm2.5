"""
NASA FIRMS Archive Downloader — ภาคเหนือประเทศไทย ปี 2023
==========================================================
ดึงข้อมูล Fire Hotspot จาก VIIRS SNPP ทั้งปี 2023
ครั้งละ 10 วัน แล้วรวมเป็นไฟล์ CSV เดียว

สิ่งที่ต้องติดตั้ง:
    pip install requests pandas tqdm

วิธีใช้:
    1. ใส่ MAP_KEY ของคุณด้านล่าง (สมัครฟรีที่ firms.modaps.eosdis.nasa.gov/api/area/)
    2. รัน: python firms_north_thailand_2023.py
"""

import requests
import pandas as pd
import time
from datetime import date, timedelta
from io import StringIO
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()  # โหลดตัวแปรจาก .env

# ══════════════════════════════════════════
# ⚙️  ตั้งค่า (แก้ตรงนี้)
# ══════════════════════════════════════════
MAP_KEY    = os.getenv("MAP_KEY")  # ← ใส่ key ของคุณในไฟล์ .env
SOURCE     = "VIIRS_SNPP_SP"     # SP = Standard Processing (archive ย้อนหลัง)
BBOX       = "97.3,17.5,101.5,20.5"  # ภาคเหนือทั้งหมด (W,S,E,N)
START_DATE = date(2023, 1, 1)
END_DATE   = date(2025, 12, 31)
CHUNK_DAYS = 5                    # FIRMS archive รับได้สูงสุด 5 วันต่อ request
OUTPUT_CSV = "firms_north2023-2025_viirs.csv"
DELAY_SEC  = 2                    # หน่วงระหว่าง request (礼貌 rate limit)
# ══════════════════════════════════════════

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def date_chunks(start: date, end: date, chunk: int):
    """แบ่งช่วงวันที่ออกเป็น chunk ขนาด chunk วัน"""
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_chunk(start: date, end: date) -> pd.DataFrame | None:
    """ดึงข้อมูลหนึ่ง chunk และคืนค่า DataFrame
    URL format ที่ถูกต้อง: /DAY_RANGE/START_DATE
    เช่น: .../5/2023-01-01  (ดึง 5 วัน ตั้งแต่ 2023-01-01)
    """
    days = (end - start).days + 1
    start_str = start.strftime('%Y-%m-%d')
    url = f"{BASE_URL}/{MAP_KEY}/{SOURCE}/{BBOX}/{days}/{start_str}"

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        text = r.text.strip()
        if not text or text.startswith("No data") or text.startswith("Invalid"):
            return None  # ไม่มีข้อมูลในช่วงนี้

        df = pd.read_csv(StringIO(text))
        return df

    except requests.exceptions.HTTPError as e:
        print(f"\n  ⚠️  HTTP Error {e.response.status_code}: {start_str} ({days} วัน)")
        return None
    except Exception as e:
        print(f"\n  ⚠️  Error: {e} — {start_str}")
        return None


def main():
    print("=" * 55)
    print("  NASA FIRMS — ภาคเหนือไทย 2023")
    print(f"  ดาวเทียม : {SOURCE}")
    print(f"  ขอบเขต   : {BBOX}")
    print(f"  ช่วงเวลา : {START_DATE} ถึง {END_DATE}")
    print("=" * 55)

    if not MAP_KEY:
        print("\n❌ กรุณาใส่ MAP_KEY ของคุณในไฟล์ .env ก่อนรัน")
        print("   สมัครฟรีได้ที่: firms.modaps.eosdis.nasa.gov/api/area/\n")
        return

    chunks = list(date_chunks(START_DATE, END_DATE, CHUNK_DAYS))
    all_dfs = []
    total_points = 0

    print(f"\nจำนวน request ทั้งหมด: {len(chunks)} ครั้ง\n")

    for start, end in tqdm(chunks, desc="ดาวน์โหลด", unit="chunk"):
        df = fetch_chunk(start, end)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            total_points += len(df)
        time.sleep(DELAY_SEC)

    if not all_dfs:
        print("\n⚠️  ไม่พบข้อมูลเลย — ตรวจสอบ MAP_KEY และ BBOX")
        return

    # รวมทุก chunk
    result = pd.concat(all_dfs, ignore_index=True)

    # แปลงประเภทข้อมูล
    result['acq_date'] = pd.to_datetime(result['acq_date'])
    result = result.sort_values('acq_date').reset_index(drop=True)

    # บันทึก
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 55}")
    print(f"  ✅  เสร็จแล้ว!")
    print(f"  จุด hotspot รวม : {total_points:,} จุด")
    print(f"  บันทึกที่        : {os.path.abspath(OUTPUT_CSV)}")
    print(f"{'=' * 55}")
    print(f"\nคอลัมน์: {list(result.columns)}")
    print(f"\nตัวอย่างข้อมูล:\n{result.head(3).to_string()}")

    # สรุปรายเดือน
    print("\n── สรุปรายเดือน ──")
    monthly = result.groupby(result['acq_date'].dt.month).size()
    month_th = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
                "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
    for m, cnt in monthly.items():
        bar = "█" * (cnt // 200)
        print(f"  {month_th[m-1]:>5}  {cnt:>6,}  {bar}")


if __name__ == "__main__":
    main()