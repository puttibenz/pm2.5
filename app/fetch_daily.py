# fetch_daily.py
# ดึงข้อมูลใหม่ประจำวัน: Open Meteo + FIRMS
# รันโดย GitHub Actions ทุกวัน 07:00 ICT (00:00 UTC)

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import time
import geopandas as gpd
from shapely.geometry import Point

load_dotenv()  # โหลดตัวแปรสภาพแวดล้อมจาก .env


def _get_with_retry(
    url: str,
    params: dict = None,
    timeout: int = 120,
    retries: int = 5,
    backoff: float = 5.0,
) -> requests.Response:
    """GET request with retry on timeout / connection errors (exponential backoff)."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == retries:
                raise
            wait = backoff * (2 ** (attempt - 1))  # 5, 10, 20, 40 …
            print(f"    Attempt {attempt} failed ({exc}), retrying in {wait:.0f}s...")
            time.sleep(wait)

# ── config ────────────────────────────────────────────────────
# พิกัดเดียวกับ src/data_collection/fetch_open_meteo.py
NORTHERN_CITIES = {
    "Chiang Mai"  : {"lat": 18.7883, "lon": 98.9853},
    "Chiang Rai"  : {"lat": 19.9105, "lon": 99.8253},
    "Mae Hong Son": {"lat": 19.3003, "lon": 97.9654},
    "Lamphun"     : {"lat": 18.5745, "lon": 99.0087},
    "Lampang"     : {"lat": 18.2888, "lon": 99.4930},
    "Phayao"      : {"lat": 19.1666, "lon": 99.9022},
    "Phrae"       : {"lat": 18.1446, "lon": 100.1403},
    "Nan"         : {"lat": 18.7756, "lon": 100.7730},
}

FIRMS_API_KEY = os.getenv("MAP_KEY")  

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = REPO_ROOT / "notebooks" / "gadm_thailand" / "gadm41_THA_1.shp"

TARGET_PROVINCES = [
    "Chiang Mai", "Chiang Rai", "Lampang", "Lamphun",
    "Mae Hong Son", "Nan", "Phayao", "Phrae",
]

TODAY = datetime.utcnow().date()
YESTERDAY = TODAY - timedelta(days=1)

# ══════════════════════════════════════════════════════════════
# 1. Open Meteo — ดึงข้อมูลอุตุนิยมวิทยา + PM2.5 ย้อนหลัง 3 วัน
# ══════════════════════════════════════════════════════════════
def fetch_open_meteo(province: str, lat: float, lon: float) -> pd.DataFrame:
    """ดึงข้อมูลรายชั่วโมงย้อนหลัง 3 วัน (buffer สำหรับ lag features)"""

    start = (TODAY - timedelta(days=3)).isoformat()
    end = TODAY.isoformat()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        # ดึง air quality (PM2.5) จาก endpoint แยก
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Bangkok",
    }

    r = _get_with_retry(url, params=params)
    data = r.json()["hourly"]

    df = pd.DataFrame(data)
    df.rename(columns={"time": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Province"] = province
    return df


def fetch_pm25_open_meteo(province: str, lat: float, lon: float) -> pd.DataFrame:
    """ดึง PM2.5 จาก air quality endpoint แยกต่างหาก"""

    start = (TODAY - timedelta(days=3)).isoformat()
    end = TODAY.isoformat()

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Bangkok",
    }

    r = _get_with_retry(url, params=params)
    data = r.json()["hourly"]

    df = pd.DataFrame({"Datetime": data["time"], "PM25": data["pm2_5"]})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Province"] = province
    return df


def fetch_all_meteo() -> pd.DataFrame:
    """รวบรวมข้อมูลทุกจังหวัด"""
    meteo_frames = []
    failed = []

    for province, coords in NORTHERN_CITIES.items():
        print(f"  Fetching meteo: {province}...")
        try:
            meteo = fetch_open_meteo(province, **coords)
            pm25 = fetch_pm25_open_meteo(province, **coords)
            combined = meteo.merge(pm25[["Datetime", "PM25", "Province"]],
                                   on=["Datetime", "Province"], how="left")
            meteo_frames.append(combined)
            time.sleep(0.5)  # rate limit
        except Exception as e:
            print(f"  ERROR {province}: {e}")
            failed.append(province)

    if not meteo_frames:
        raise RuntimeError(
            f"fetch_all_meteo: ทุกจังหวัดล้มเหลว ({', '.join(failed)}). "
            "ตรวจสอบ network / API endpoint แล้วลองใหม่"
        )
    if failed:
        print(f"  WARNING: ข้อมูลไม่ครบ — ขาดจังหวัด: {', '.join(failed)}")

    return pd.concat(meteo_frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════
# 2. NASA FIRMS — ดึง Fire Hotspot วันล่าสุด
# ══════════════════════════════════════════════════════════════
def fetch_firms(days_back: int = 3) -> pd.DataFrame:
    """
    ดึง VIIRS hotspot ย้อนหลัง N วัน
    bounding box ครอบภาคเหนือไทย: 17-21N, 97-102E
    """
    # FIRMS API: CSV format, ระบุ bbox และจำนวนวัน
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        f"/{FIRMS_API_KEY}/VIIRS_SNPP_NRT"
        f"/97.3,17.5,101.5,20.5"  # west,south,east,north (ครอบภาคเหนือ)
        f"/{days_back}"           # ย้อนหลังกี่วัน
    )

    r = _get_with_retry(url, timeout=60)

    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    # clean เหมือนขั้นตอนก่อนหน้า
    mask = df["confidence"] != "l"
    if "type" in df.columns:
        mask = mask & (df["type"] == 0)
    else:
        print("  WARNING: 'type' column missing from FIRMS response, skipping type filter")
    df = df[mask].copy()

    print(f"  FIRMS: {len(df)} hotspots (last {days_back} days)")
    return df


def aggregate_firms_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spatial join lat/lon → จังหวัด แล้ว aggregate เป็น daily × Province
    คืนค่า DataFrame ที่มีคอลัมน์: date, Province, hotspot_count, frp_sum, frp_mean
    """
    # โหลด shapefile ภาคเหนือ 8 จังหวัด
    thailand = gpd.read_file(SHP_PATH)
    north = thailand[thailand["NAME_1"].isin(TARGET_PROVINCES)].copy()
    north["Province"] = north["NAME_1"]
    north = north.to_crs("EPSG:4326")

    # แปลง FIRMS เป็น GeoDataFrame
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    firms_geo = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Point-in-polygon join
    joined = gpd.sjoin(
        firms_geo,
        north[["Province", "geometry"]],
        how="left",
        predicate="within",
    )

    outside = joined["Province"].isna().sum()
    print(f"  Points outside all provinces: {outside:,} ({outside/len(joined)*100:.1f}%)")

    joined = joined.dropna(subset=["Province"])
    print(f"  Points matched to province: {len(joined):,}")

    # Aggregate รายวัน × จังหวัด
    joined["acq_date"] = pd.to_datetime(joined["acq_date"])
    hotspot_daily = (
        joined
        .groupby(["acq_date", "Province"])
        .agg(
            hotspot_count=("frp", "count"),
            frp_sum=("frp", "sum"),
            frp_mean=("frp", "mean"),
        )
        .reset_index()
        .rename(columns={"acq_date": "date"})
    )

    # Fill zero-fire days
    all_dates = pd.date_range(
        start=joined["acq_date"].min(),
        end=joined["acq_date"].max(),
        freq="D",
    )
    full_grid = pd.MultiIndex.from_product(
        [all_dates, TARGET_PROVINCES],
        names=["date", "Province"],
    ).to_frame(index=False)

    hotspot_daily = full_grid.merge(hotspot_daily, on=["date", "Province"], how="left")
    hotspot_daily[["hotspot_count", "frp_sum", "frp_mean"]] = (
        hotspot_daily[["hotspot_count", "frp_sum", "frp_mean"]].fillna(0)
    )

    print(f"  FIRMS daily shape: {hotspot_daily.shape}")
    return hotspot_daily


# ══════════════════════════════════════════════════════════════
# 3. Append to master CSV (ไม่ overwrite ข้อมูลเก่า)
# ══════════════════════════════════════════════════════════════
def append_to_master(new_df: pd.DataFrame, path: Path, date_col: str):
    """
    ต่อท้ายข้อมูลใหม่เข้าไฟล์ master
    ถ้ามีวันนั้นอยู่แล้ว → ข้าม (idempotent)
    """
    if path.exists():
        master = pd.read_csv(path, parse_dates=[date_col])
        existing_dates = set(master[date_col].dt.date)
        new_dates = set(pd.to_datetime(new_df[date_col]).dt.date)
        truly_new = new_df[
            pd.to_datetime(new_df[date_col]).dt.date.apply(
                lambda d: d not in existing_dates
            )
        ]
        if truly_new.empty:
            print(f"  {path.name}: ไม่มีข้อมูลใหม่ ข้ามไป")
            return
        combined = pd.concat([master, truly_new], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    print(f"  Saved → {path.name} ({len(combined)} rows total)")


# ══════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Starting daily fetch...")

    # Open Meteo
    print("\nOpen Meteo + PM2.5:")
    meteo_df = fetch_all_meteo()
    append_to_master(
        meteo_df,
        RAW_DATA_DIR / "openmeteo_all_provinces.csv",
        date_col="Datetime"
    )

    # FIRMS
    print("\nNASA FIRMS:")
    firms_raw = fetch_firms(days_back=3)
    firms_df = aggregate_firms_daily(firms_raw)
    append_to_master(
        firms_df,
        PROCESSED_DATA_DIR / "firms_daily_by_province.csv",
        date_col="date"
    )

    print("\nDone.")