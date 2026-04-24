"""ส่วนประกอบย่อยสำหรับ Streamlit Dashboard"""

import streamlit as st
import plotly.express as px
import pandas as pd


def pm25_level_indicator(pm25_value):
    """แสดงระดับคุณภาพอากาศตามค่า PM2.5"""
    if pm25_value <= 25:
        return "ดีมาก", "🟢"
    elif pm25_value <= 37:
        return "ดี", "🟡"
    elif pm25_value <= 50:
        return "ปานกลาง", "🟠"
    elif pm25_value <= 90:
        return "เริ่มมีผลต่อสุขภาพ", "🔴"
    else:
        return "มีผลต่อสุขภาพ", "🟣"


def plot_pm25_trend(df, date_col="time", pm25_col="pm25"):
    """พล็อตกราฟแนวโน้ม PM2.5"""
    fig = px.line(
        df,
        x=date_col,
        y=pm25_col,
        title="แนวโน้ม PM2.5",
        labels={pm25_col: "PM2.5 (µg/m³)", date_col: "วันที่"},
    )
    fig.add_hline(y=37, line_dash="dash", line_color="orange",
                  annotation_text="มาตรฐาน (37 µg/m³)")
    return fig


def plot_hotspot_map(df, lat_col="latitude", lon_col="longitude"):
    """พล็อตแผนที่จุดความร้อน"""
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color_continuous_scale="YlOrRd",
        zoom=8,
        mapbox_style="open-street-map",
        title="จุดความร้อน (Hotspot) บริเวณเชียงใหม่",
    )
    return fig
