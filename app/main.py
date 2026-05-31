"""Streamlit Dashboard — PM2.5 Early Warning System, Northern Thailand"""

import sys
import os
# เพิ่ม root และ app directory เข้า path เพื่อให้ import ได้ทุกสภาพแวดล้อม
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import textwrap
import streamlit as st
import plotly.express as px
from pathlib import Path

from components import (
    PROVINCE_COORDS,
    clean_html,
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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Prompt:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply Font Globally */
    html, body, [data-testid="stAppViewContainer"], .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
        font-family: 'Prompt', 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Modern Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(241, 245, 249, 0.5);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.4);
        border-radius: 9999px;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 116, 139, 0.6);
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    /* Clean custom sidebar style */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Elegant tabs override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        border-bottom: 2px solid rgba(226, 232, 240, 0.8) !important;
        background-color: transparent !important;
        padding-top: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Prompt', 'Plus Jakarta Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        background-color: transparent !important;
        border-radius: 10px 10px 0px 0px !important;
        padding: 12px 20px !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: -2px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f62fe !important;
        background-color: rgba(15, 98, 254, 0.04) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0f62fe !important;
        background-color: rgba(15, 98, 254, 0.02) !important;
        border-bottom: 3px solid #0f62fe !important;
    }
    
    /* Calendar scroll list */
    .cal-grid {
        display: flex;
        overflow-x: auto;
        gap: 16px;
        padding: 8px 4px 16px 4px;
        scroll-behavior: smooth;
    }
    .cal-item {
        min-width: 125px;
        flex: 0 0 auto;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .cal-item:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 24px -8px rgba(15, 98, 254, 0.15) !important;
        border-color: rgba(15, 98, 254, 0.3) !important;
    }
    
    /* Premium Cards base style */
    .premium-card {
        background: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 20px -2px rgba(148, 163, 184, 0.06), 0 2px 8px -1px rgba(148, 163, 184, 0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 145px;
        position: relative;
        overflow: hidden;
    }
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 36px -6px rgba(15, 98, 254, 0.12), 0 4px 12px -2px rgba(15, 98, 254, 0.05);
        border-color: rgba(15, 98, 254, 0.2);
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
        
        # Try to load feature list from JSON (PRD BUG-03)
        feat_path = ROOT / "artifacts" / "feature_list.json"
        if feat_path.exists():
            import json
            with open(feat_path, encoding='utf-8') as f:
                feature_names = json.load(f)
        else:
            feature_names = list(model.feature_names_in_)
            
        return model, feature_names
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

actual_data = prov_data[prov_data["is_predicted"] == False]
latest = actual_data.iloc[-1] if not actual_data.empty else prov_data.iloc[0]
fore_period = prov_data[prov_data["is_predicted"] == True].head(168)

pm25_now = float(latest["PM25"])
pm25_pred_max = float(fore_period["predicted"].max()) if not fore_period.empty else pm25_now
pm25_trend = float(fore_period["predicted"].mean() - actual_data.tail(168)["PM25"].mean()) if not fore_period.empty and not actual_data.empty else 0.0
wind_deg = float(latest.get("wind_direction_10m", 0))

st.markdown(
    clean_html(f"""
    <div style="background: linear-gradient(135deg, #0f62fe 0%, #1d4ed8 50%, #1e40af 100%);
                padding: 30px 35px;
                border-radius: 20px;
                box-shadow: 0 10px 30px -10px rgba(15, 98, 254, 0.25);
                margin-bottom: 28px;
                color: #ffffff;
                position: relative;
                overflow: hidden;">
        <!-- Decorative subtle background glow -->
        <div style="position: absolute; top: -50%; right: -15%; width: 260px; height: 260px; 
                    background: rgba(255, 255, 255, 0.12); border-radius: 50%; filter: blur(45px); pointer-events: none;"></div>
        
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 42px; background: rgba(255, 255, 255, 0.15); width: 76px; height: 76px; 
                        display: flex; align-items: center; justify-content: center; border-radius: 16px; 
                        backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.25); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);">
                🌫️
            </div>
            <div>
                <h1 style="margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; font-family: 'Prompt', sans-serif; color: #ffffff; line-height: 1.2;">
                    ระบบเตือนภัยล่วงหน้า PM2.5 — จังหวัด{province}
                </h1>
                <p style="margin: 8px 0 0 0; font-size: 14px; opacity: 0.9; font-weight: 500; font-family: 'Prompt', sans-serif;">
                    ข้อมูล ณ เวลา: <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; background: rgba(255,255,255,0.15); padding: 2px 8px; border-radius: 6px; margin-right: 4px;">{latest['Datetime'].strftime('%d %b %Y %H:%M')}</span> ICT | ระบบประมวลผลสภาพอากาศอัจฉริยะ 8 จังหวัดภาคเหนือ
                </p>
            </div>
        </div>
    </div>
    """),
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)
prev_24h_data = prov_data[prov_data["Datetime"] <= now - pd.Timedelta(days=1)]
prev_24h = prev_24h_data.iloc[-1] if not prev_24h_data.empty else latest
recent_firms = firms[firms["acq_date"] >= firms["acq_date"].max() - pd.Timedelta(days=1)]
info_now = pm25_level_info(pm25_now)
info_fore = pm25_level_info(pm25_pred_max)

diff_24h = pm25_now - float(prev_24h['PM25'])
diff_color = "#ef4444" if diff_24h > 0 else "#10b981"
diff_text = f"{diff_24h:+.1f} จาก 24h ก่อน"

with k1:
    st.markdown(
        clean_html(f"""
        <div class="premium-card" style="border-left: 5px solid {info_now['color']};">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: #64748b; font-family: 'Prompt', sans-serif;">PM2.5 ปัจจุบัน</span>
            <span style="font-size: 20px; background: {info_now['color']}15; padding: 4px 8px; border-radius: 8px; line-height: 1;">{info_now['emoji']}</span>
          </div>
          <div style="margin-top: 12px;">
            <span style="font-size: 28px; font-weight: 800; color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: -0.5px;">{pm25_now:.1f}</span>
            <span style="font-size: 13px; font-weight: 600; color: #64748b; font-family: 'Prompt', sans-serif; margin-left: 4px;">µg/m³</span>
          </div>
          <div style="font-size: 12px; font-weight: 700; color: {diff_color}; margin-top: 10px; display: flex; align-items: center; gap: 4px; font-family: 'Prompt', sans-serif;">
            <span>{diff_text}</span>
          </div>
        </div>
        """),
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        clean_html(f"""
        <div class="premium-card" style="border-left: 5px solid {info_fore['color']};">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: #64748b; font-family: 'Prompt', sans-serif;">สูงสุดพยากรณ์ 7 วัน</span>
            <span style="font-size: 20px; background: {info_fore['color']}15; padding: 4px 8px; border-radius: 8px; line-height: 1;">{info_fore['emoji']}</span>
          </div>
          <div style="margin-top: 12px;">
            <span style="font-size: 28px; font-weight: 800; color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: -0.5px;">{pm25_pred_max:.1f}</span>
            <span style="font-size: 13px; font-weight: 600; color: #64748b; font-family: 'Prompt', sans-serif; margin-left: 4px;">µg/m³</span>
          </div>
          <div style="font-size: 11px; font-weight: 700; color: {info_fore['color']}; background: {info_fore['color']}12; padding: 2px 8px; border-radius: 6px; display: inline-block; margin-top: 8px; font-family: 'Prompt', sans-serif;">
            {info_fore['label']}
          </div>
        </div>
        """),
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        clean_html(f"""
        <div class="premium-card" style="border-left: 5px solid #ff7043;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: #64748b; font-family: 'Prompt', sans-serif;">จุดความร้อน (24h ล่าสุด)</span>
            <span style="font-size: 20px; background: #ff704315; padding: 4px 8px; border-radius: 8px; line-height: 1;">🔥</span>
          </div>
          <div style="margin-top: 12px;">
            <span style="font-size: 28px; font-weight: 800; color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: -0.5px;">{len(recent_firms):,}</span>
            <span style="font-size: 13px; font-weight: 600; color: #64748b; font-family: 'Prompt', sans-serif; margin-left: 4px;">จุด</span>
          </div>
          <div style="font-size: 12px; font-weight: 600; color: #475569; margin-top: 10px; font-family: 'Prompt', sans-serif;">
            FRP รวม <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #ff7043;">{recent_firms['frp'].sum():.0f}</span> MW
          </div>
        </div>
        """),
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        clean_html(f"""
        <div class="premium-card" style="border-left: 5px solid #0f62fe;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: #64748b; font-family: 'Prompt', sans-serif;">อุณหภูมิ / ความชื้น</span>
            <span style="font-size: 20px; background: #0f62fe15; padding: 4px 8px; border-radius: 8px; line-height: 1;">🌡️</span>
          </div>
          <div style="margin-top: 12px;">
            <span style="font-size: 28px; font-weight: 800; color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: -0.5px;">{latest.get('temperature_2m', 0):.1f}</span>
            <span style="font-size: 13px; font-weight: 600; color: #64748b; font-family: 'Prompt', sans-serif; margin-left: 4px;">°C</span>
          </div>
          <div style="font-size: 12px; font-weight: 600; color: #475569; margin-top: 10px; font-family: 'Prompt', sans-serif;">
            ความชื้น <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0f62fe;">{latest.get('relative_humidity_2m', 0):.0f}%</span>
          </div>
        </div>
        """),
        unsafe_allow_html=True
    )

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
            st.markdown(clean_html(f"""
                <div class="cal-item" style="background: rgba(255, 255, 255, 0.85);
                             border: 1px solid rgba(226, 232, 240, 0.8);
                             border-top: 4px solid {info['color']};
                             text-align: center;
                             padding: 16px 12px;
                             border-radius: 16px;
                             box-shadow: 0 4px 12px -2px rgba(148, 163, 184, 0.05);
                             min-width: 120px;
                             backdrop-filter: blur(8px);
                             font-family: 'Prompt', sans-serif;">
                  <div style="font-size: 28px; margin-bottom: 8px; line-height: 1;">{info["emoji"]}</div>
                  <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; font-family: 'Prompt', sans-serif; letter-spacing: 0.5px;">{row["date"].strftime("%d %b")}</div>
                  <div style="font-weight: 800; color: #1e293b; font-size: 20px; margin: 6px 0; font-family: 'Plus Jakarta Sans', sans-serif;">{row["predicted"]:.0f}</div>
                  <div style="font-size: 10px; font-weight: 800; color: {info['color']}; background: {info['color']}12; padding: 3px 8px; border-radius: 6px; display: inline-block; font-family: 'Prompt', sans-serif; border: 1px solid {info['color']}20;">
                    {info["label"]}
                  </div>
                </div>
                """), unsafe_allow_html=True)
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
    model, feature_names = load_model_artifacts()
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
