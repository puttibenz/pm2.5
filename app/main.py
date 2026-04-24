"""Streamlit Dashboard สำหรับระบบเตือนภัย PM2.5 เชียงใหม่"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="PM2.5 Early Warning - เชียงใหม่",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ ระบบเตือนภัยล่วงหน้า PM2.5 - เชียงใหม่")
st.markdown("---")


@st.cache_resource
def load_model():
    """โหลดโมเดลที่เทรนแล้ว"""
    model_path = os.path.join("app", "saved_models", "xgboost_pm25.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# Sidebar
st.sidebar.header("ตั้งค่า")
st.sidebar.info("ระบบเตือนภัยล่วงหน้าค่าฝุ่น PM2.5 สำหรับจังหวัดเชียงใหม่")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="PM2.5 ปัจจุบัน", value="-- µg/m³", delta="-- จากเมื่อวาน")
with col2:
    st.metric(label="พยากรณ์พรุ่งนี้", value="-- µg/m³")
with col3:
    st.metric(label="ระดับคุณภาพอากาศ", value="--")

st.markdown("---")

# TODO: เพิ่มกราฟแสดงแนวโน้ม PM2.5
st.subheader("📈 แนวโน้ม PM2.5 (7 วันล่าสุด)")
st.info("กรุณาเตรียมข้อมูลและเทรนโมเดลก่อนใช้งาน Dashboard")

# TODO: เพิ่มแผนที่แสดงจุดความร้อน
st.subheader("🗺️ แผนที่จุดความร้อน (Hotspot)")
st.info("เชื่อมต่อข้อมูล NASA FIRMS เพื่อแสดงแผนที่")

# TODO: เพิ่มตารางข้อมูลสภาพอากาศ
st.subheader("🌤️ ข้อมูลสภาพอากาศ")
st.info("เชื่อมต่อข้อมูล Open-Meteo เพื่อแสดงสภาพอากาศ")
