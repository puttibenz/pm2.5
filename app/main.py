"""Streamlit Dashboard — PM2.5 Early Warning System, Northern Thailand"""

import sys
import os
# เพิ่ม root และ app directory เข้า path เพื่อให้ import ได้ทุกสภาพแวดล้อม
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from pathlib import Path

from components import (
    PROVINCE_COORDS,
    get_shap_summary_html,
    plot_7day_forecast,
    plot_feature_importance,
    plot_hotspot_priority_map,
    plot_shap_waterfall,
    pm25_level_info,
    render_alert_section,
    render_province_overview,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PM2.5 Early Warning — ภาคเหนือ",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent

# ─── Session State ───────────────────────────────────────────────────────────
if "selected_province" not in st.session_state:
    st.session_state["selected_province"] = "Chiang Mai"

# ─── CSS Overrides ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"]  { font-size: 1.5rem; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem; }
    .stTabs [data-baseweb="tab"]   { font-size: 0.95rem; font-weight: 600; }
    /* Horizontal scroll for calendar on small screens */
    .cal-grid {
        display: flex;
        overflow-x: auto;
        gap: 10px;
        padding-bottom: 10px;
    }
    .cal-item {
        min-width: 100px;
        flex: 0 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Data Loading (Cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ กำลังโหลดโมเดล...")
def load_model_artifacts():
    try:
        model  = joblib.load(ROOT / "artifacts" / "xgboost_pm25.pkl")
        scaler = joblib.load(ROOT / "artifacts" / "scaler.pkl")
        
        # Try to load feature list from JSON (PRD BUG-03)
        feat_path = ROOT / "artifacts" / "feature_list.json"
        if feat_path.exists():
            import json
            with open(feat_path, encoding='utf-8') as f:
                feature_names = json.load(f)
        else:
            feature_names = list(model.feature_names_in_)
            
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลด Model Artifacts ได้: {e}")
        st.stop()


@st.cache_data(show_spinner="📂 กำลังโหลดข้อมูล...", ttl=3600)
def load_dashboard_data() -> pd.DataFrame:
    """
    อ่าน dashboard_data.csv ที่ predict.py สร้างไว้แล้ว
    """
    path = ROOT / "data" / "processed" / "dashboard_data.csv"
    if not path.exists():
        st.error(
            "⚠️ ยังไม่มีไฟล์ `dashboard_data.csv`\n\n"
            "รัน `python app/predict.py` ก่อนเพื่อสร้างข้อมูลพยากรณ์"
        )
        st.stop()
    return pd.read_csv(path, parse_dates=["Datetime"])


@st.cache_data(show_spinner="🛰️ กำลังโหลดข้อมูล FIRMS...", ttl=3600)
def load_firms_data() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "firms_recent_hotspots.csv"
    if not path.exists():
        return pd.DataFrame(columns=["acq_date", "latitude", "longitude", "frp"])
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


@st.cache_data(show_spinner="🧠 กำลังโหลด SHAP values...", ttl=3600)
def load_shap_data() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "shap_latest.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
all_data = load_dashboard_data()

with st.sidebar:
    st.title("🌫️ PM2.5 Early Warning")
    st.caption("ระบบเตือนภัยล่วงหน้าฝุ่น PM2.5 ภาคเหนือ")
    st.divider()

    province = st.selectbox(
        "📍 เลือกจังหวัด",
        sorted(PROVINCE_COORDS.keys()),
        index=sorted(PROVINCE_COORDS.keys()).index(st.session_state["selected_province"]),
        key="province_selector"
    )
    if province != st.session_state["selected_province"]:
        st.session_state["selected_province"] = province
        st.rerun()

    st.divider()
    st.markdown("**📡 แหล่งข้อมูล**")
    st.caption("🛰️ ดาวเทียม: NASA FIRMS VIIRS")
    st.caption("🌤️ สภาพอากาศ: Open-Meteo API")
    st.caption("🤖 โมเดล: XGBoost (69 features)")
    st.divider()
    
    last_update = all_data['Datetime'].max()
    st.caption(f"อัปเดตล่าสุด: {last_update.strftime('%d %b %Y %H:%M')} ICT")

# ─── Main Content ─────────────────────────────────────────────────────────────
prov_data = all_data[all_data["Province"] == province].sort_values("Datetime").reset_index(drop=True)
firms = load_firms_data()
now = pd.Timestamp.now(tz='Asia/Bangkok').tz_localize(None).floor('h')

actual_data = prov_data[prov_data["Datetime"] <= now]
latest = actual_data.iloc[-1] if not actual_data.empty else prov_data.iloc[0]
fore_period = prov_data[prov_data["Datetime"] > now].head(168)

pm25_now = float(latest["PM25"])
pm25_pred_max = float(fore_period["predicted"].max()) if not fore_period.empty else pm25_now
pm25_trend = float(fore_period["predicted"].mean() - actual_data.tail(168)["PM25"].mean()) if not fore_period.empty and not actual_data.empty else 0.0
wind_deg = float(latest.get("wind_direction_10m", 0))

st.markdown(f"# 🌫️ ระบบเตือนภัยล่วงหน้า PM2.5 — {province}")
st.caption(f"ข้อมูลปัจจุบัน: {latest['Datetime'].strftime('%d %b %Y %H:%M')} ICT")

k1, k2, k3, k4 = st.columns(4)
with k1:
    prev_24h_data = prov_data[prov_data["Datetime"] <= now - pd.Timedelta(days=1)]
    prev_24h = prev_24h_data.iloc[-1] if not prev_24h_data.empty else latest
    st.metric(label=f"PM2.5 ปัจจุบัน  {pm25_level_info(pm25_now)['emoji']}", value=f"{pm25_now:.1f} µg/m³", delta=f"{pm25_now - float(prev_24h['PM25']):+.1f} จาก 24h ก่อน", delta_color="inverse")
with k2:
    info_fore = pm25_level_info(pm25_pred_max)
    st.metric(label=f"สูงสุดพยากรณ์ 7 วัน  {info_fore['emoji']}", value=f"{pm25_pred_max:.1f} µg/m³", delta=info_fore["label"], delta_color="inverse")
with k3:
    recent_firms = firms[firms["acq_date"] >= firms["acq_date"].max() - pd.Timedelta(days=1)]
    st.metric(label="🔥 จุดความร้อน (24h ล่าสุด)", value=f"{len(recent_firms):,} จุด", delta=f"FRP รวม {recent_firms['frp'].sum():.0f} MW", delta_color="off")
with k4:
    st.metric(label="🌡️ อุณหภูมิ / ความชื้น", value=f"{latest.get('temperature_2m', 0):.1f} °C", delta=f"ความชื้น {latest.get('relative_humidity_2m', 0):.0f}%", delta_color="off")

st.divider()

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  ภาพรวม 8 จังหวัด", "🔮  พยากรณ์ 7 วัน", "🚨  ระบบแจ้งเตือน", "🗺️  แผนที่จุดเสี่ยง", "🧠  อธิบายโมเดล (SHAP)", "📈  Model Performance"
])

with tab0:
    render_province_overview(all_data, now)

with tab1:
    st.plotly_chart(plot_7day_forecast(prov_data, province, now=now), use_container_width=True)
    st.subheader("📋 สรุปพยากรณ์รายวัน (7 วันข้างหน้า)")
    if not fore_period.empty:
        daily_rows = fore_period.copy()
        daily_rows["date_obj"] = daily_rows["Datetime"].dt.date
        daily_tbl = daily_rows.groupby("date_obj")["predicted"].agg(["mean", "max", "min"]).reset_index()
        daily_tbl["วันที่"] = pd.to_datetime(daily_tbl["date_obj"]).dt.strftime("%d %b %Y")
        daily_tbl["ระดับ"] = daily_tbl["max"].apply(lambda x: f"{pm25_level_info(x)['emoji']} {pm25_level_info(x)['label']}")
        st.dataframe(daily_tbl[["วันที่", "mean", "max", "min", "ระดับ"]].round(1), use_container_width=True, hide_index=True)

with tab2:
    render_alert_section(pm25_pred_max, province, pm25_trend)
    st.divider()
    st.subheader("📅 ปฏิทินความเสี่ยง (7 วันข้างหน้า)")
    if not fore_period.empty:
        daily_cal = fore_period.copy()
        daily_cal["date"] = daily_cal["Datetime"].dt.date
        daily_max = daily_cal.groupby("date")["predicted"].max().reset_index()
        st.markdown('<div class="cal-grid">', unsafe_allow_html=True)
        for _, row in daily_max.iterrows():
            info = pm25_level_info(row["predicted"])
            st.markdown(f"""<div class="cal-item" style='background:{info["color"]}1a;border:1px solid {info["color"]}44;text-align:center;padding:12px 6px;border-radius:8px'>
                <div style='font-size:26px'>{info["emoji"]}</div><div style='font-size:11px;color:#666'>{row["date"].strftime("%d %b")}</div>
                <div style='font-weight:bold;color:{info["color"]};font-size:16px'>{row["predicted"]:.0f}</div>
                <div style='font-size:10px;color:#555'>{info["label"]}</div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    col_map, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.subheader("⚙️ ตั้งค่า")
        days_back = st.slider("ย้อนหลัง (วัน)", 1, 90, 30)
        top_n = st.slider("Top N จุดสำคัญ", 5, 50, 20)
        wind_override = st.number_input("ทิศทางลม (องศา)", 0.0, 360.0, float(wind_deg), step=5.0)
    with col_map:
        recent_firms = firms[firms["acq_date"] >= firms["acq_date"].max() - pd.Timedelta(days=days_back)]
        st.plotly_chart(plot_hotspot_priority_map(recent_firms, province, wind_override, top_n), use_container_width=True)

with tab4:
    st.subheader("🧠 อธิบายการตัดสินใจของโมเดล (SHAP)")
    model, scaler, feature_names = load_model_artifacts()
    shap_df = load_shap_data()
    X_latest_full = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for f in feature_names:
        if f in prov_data.columns: X_latest_full.at[0, f] = float(prov_data[f].iloc[-1])
    pm25_latest_pred = float(prov_data["predicted"].iloc[-1])
    summary_html = get_shap_summary_html(model, X_latest_full, feature_names, pm25_latest_pred, precomputed_shap=shap_df, province=province)
    if summary_html: st.markdown(f"<div style='background:#e3f2fd;border-left:4px solid #42a5f5;padding:14px;border-radius:8px;margin-bottom:16px;color:#1565c0'>{summary_html}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: 
        wf_fig = plot_shap_waterfall(model, X_latest_full, feature_names, precomputed_shap=shap_df, province=province)
        if wf_fig:
            st.plotly_chart(wf_fig, use_container_width=True)
        else:
            st.warning("ไม่สามารถแสดง SHAP Waterfall ได้")
    with c2: st.plotly_chart(plot_feature_importance(model, feature_names), use_container_width=True)

with tab5:
    st.subheader("📈 การติดตามความแม่นยำของโมเดล (MAE / RMSE)")
    try:
        hist_path, actual_path = ROOT / "predictions" / "predictions_history.csv", ROOT / "data" / "raw" / "openmeteo_all_provinces.csv"
        if hist_path.exists() and actual_path.exists():
            h_df, a_df = pd.read_csv(hist_path, parse_dates=["predict_datetime"]), pd.read_csv(actual_path, parse_dates=["Datetime"])
            perf = h_df.merge(a_df[["Datetime", "Province", "PM25"]], left_on=["predict_datetime", "Province"], right_on=["Datetime", "Province"])
            if not perf.empty:
                perf["error"] = (perf["pred_24h"] - perf["PM25"]).abs()
                m1, m2 = st.columns(2)
                m1.metric("MAE (24h Forecast)", f"{perf['error'].mean():.2f} µg/m³")
                m2.metric("Samples", len(perf))
                st.plotly_chart(px.histogram(perf, x="error", color="Province", title="การกระจายตัวของความคลาดเคลื่อน (Absolute Error)"), use_container_width=True)
            else: st.info("ยังไม่มีข้อมูลจริงมาเทียบกับประวัติพยากรณ์")
        else: st.info("ไม่พบไฟล์ประวัติพยากรณ์หรือข้อมูลจริง")
    except Exception as e: st.error(f"เกิดข้อผิดพลาด: {e}")
