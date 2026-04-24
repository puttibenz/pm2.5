"""สร้าง Features สำหรับโมเดล (Lag, Rolling Average, Merge ข้อมูล)"""

import os
import pandas as pd
import numpy as np


def load_raw_data():
    """โหลดข้อมูลดิบจากโฟลเดอร์ data/raw/"""
    air4thai = pd.read_csv(os.path.join("data", "raw", "air4thai_data.csv"))
    meteo = pd.read_csv(os.path.join("data", "raw", "open_meteo_data.csv"))
    firms = pd.read_csv(os.path.join("data", "raw", "nasa_firms_data.csv"))
    return air4thai, meteo, firms


def clean_data(df):
    """จัดการ Missing Values"""
    # TODO: ปรับแต่งตามโครงสร้างข้อมูลจริง
    df = df.dropna(subset=["time"]) if "time" in df.columns else df
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def create_lag_features(df, column="pm25", lags=[1, 2, 3, 7]):
    """สร้าง Lag Features"""
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def create_rolling_features(df, column="pm25", windows=[3, 7, 14]):
    """สร้าง Rolling Average Features"""
    for window in windows:
        df[f"{column}_rolling_mean_{window}"] = (
            df[column].rolling(window=window).mean()
        )
        df[f"{column}_rolling_std_{window}"] = (
            df[column].rolling(window=window).std()
        )
    return df


def build_features():
    """Pipeline หลักสำหรับสร้าง Features"""
    print("กำลังโหลดข้อมูลดิบ...")
    air4thai, meteo, firms = load_raw_data()

    print("กำลัง Clean ข้อมูล...")
    air4thai = clean_data(air4thai)
    meteo = clean_data(meteo)

    # TODO: Merge ข้อมูลจากแหล่งต่างๆ เข้าด้วยกัน
    # df = pd.merge(air4thai, meteo, on="time", how="left")

    # TODO: สร้าง Features
    # df = create_lag_features(df)
    # df = create_rolling_features(df)

    # TODO: บันทึกข้อมูลที่ทำ Feature Engineering แล้ว
    # output_path = os.path.join("data", "features", "features_dataset.csv")
    # df.to_csv(output_path, index=False)
    # print(f"บันทึก Features สำเร็จ: {output_path}")

    print("Feature Engineering เสร็จสิ้น (TODO: ต้องปรับแต่งตามข้อมูลจริง)")


if __name__ == "__main__":
    build_features()
