"""Streamlit Dashboard — PM2.5 Early Warning System, Northern Thailand"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

from components import (
    PROVINCE_COORDS,
    build_province_features,
    get_shap_summary_html,
    plot_72h_forecast,
    plot_feature_importance,
    plot_hotspot_priority_map,
    plot_shap_waterfall,
    pm25_level_info,
    render_alert_section,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PM2.5 Early Warning — ภาคเหนือ",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent

# ─── CSS Overrides ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"]  { font-size: 1.5rem; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem; }
    .stTabs [data-baseweb="tab"]   { font-size: 0.95rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Data Loading (Cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ กำลังโหลดโมเดล...")
def load_model_artifacts():
    model  = joblib.load(ROOT / "artifacts" / "xgboost_pm25.pkl")
    scaler = joblib.load(ROOT / "artifacts" / "scaler.pkl")
    return model, scaler, list(model.feature_names_in_)


@st.cache_data(show_spinner="📂 กำลังโหลดข้อมูลประวัติ...")
def load_merged_data() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "processed" / "openmeteo_firms_merged.csv",
        parse_dates=["Datetime"],
    )


@st.cache_data(show_spinner="🛰️ กำลังโหลดข้อมูล FIRMS...")
def load_firms_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "raw" / "firms_north2023-2025_viirs.csv")
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


@st.cache_data(show_spinner="🔮 กำลังพยากรณ์...", ttl=3600)
def get_province_predictions(province: str) -> pd.DataFrame:
    """Build feature matrix + run model predictions for one province (cached 1 h)."""
    merged = load_merged_data()
    model, _, feature_names = load_model_artifacts()

    pte     = merged.groupby("Province")["PM25"].mean().to_dict()
    prov_df = merged[merged["Province"] == province].copy()
    feat_df = build_province_features(prov_df, province, pte)

    X = feat_df[feature_names].fillna(0)
    feat_df = feat_df.copy()
    feat_df["predicted"] = model.predict(X)

    keep = list(dict.fromkeys(
        ["Datetime", "Province", "PM25", "predicted",
         "wind_direction_10m", "temperature_2m", "relative_humidity_2m",
         "hotspot_count", "frp_sum"]
        + feature_names
    ))
    return feat_df[[c for c in keep if c in feat_df.columns]]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ PM2.5 Early Warning")
    st.caption("ระบบเตือนภัยล่วงหน้าฝุ่น PM2.5 ภาคเหนือ")
    st.divider()

    province = st.selectbox(
        "📍 เลือกจังหวัด",
        sorted(PROVINCE_COORDS.keys()),
        index=sorted(PROVINCE_COORDS.keys()).index("Chiang Mai"),
    )

    st.divider()
    st.markdown("**📡 แหล่งข้อมูล**")
    st.caption("🛰️ ดาวเทียม: NASA FIRMS VIIRS")
    st.caption("🌤️ สภาพอากาศ: Open-Meteo API")
    st.caption("🤖 โมเดล: XGBoost (69 features)")
    st.divider()
    st.caption("อัปเดตอัตโนมัติทุกวัน เวลา 07:00 ICT")

# ─── Load Data ────────────────────────────────────────────────────────────────
model, scaler, feature_names = load_model_artifacts()

with st.spinner(f"🔮 พยากรณ์ {province}..."):
    prov_data = get_province_predictions(province)

firms = load_firms_data()

# Derived values for KPI cards
latest    = prov_data.sort_values("Datetime").iloc[-1]
prev_24h  = prov_data.sort_values("Datetime").iloc[-25] if len(prov_data) > 25 else latest
fore72    = prov_data.tail(72)

pm25_now     = float(latest["PM25"])
pm25_pred72  = float(fore72["predicted"].max())
pm25_trend   = (
    float(fore72["predicted"].mean() - prov_data.iloc[-97:-25]["predicted"].mean())
    if len(prov_data) > 97 else 0.0
)
wind_deg = float(latest.get("wind_direction_10m", 0))

info_now  = pm25_level_info(pm25_now)
info_fore = pm25_level_info(pm25_pred72)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"# 🌫️ ระบบเตือนภัยล่วงหน้า PM2.5 — {province}")
st.caption(
    f"ข้อมูลล่าสุด: {latest['Datetime'].strftime('%d %b %Y %H:%M')} ICT  "
    f"| อัปเดตอัตโนมัติทุกวัน"
)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    delta_pm25 = pm25_now - float(prev_24h["PM25"])
    st.metric(
        label=f"PM2.5 ปัจจุบัน  {info_now['emoji']}",
        value=f"{pm25_now:.1f} µg/m³",
        delta=f"{delta_pm25:+.1f} จาก 24h ก่อน",
        delta_color="inverse",
    )

with k2:
    st.metric(
        label=f"สูงสุดพยากรณ์ 72h  {info_fore['emoji']}",
        value=f"{pm25_pred72:.1f} µg/m³",
        delta=info_fore["label"],
        delta_color="off",
    )

with k3:
    recent_firms = firms[
        firms["acq_date"] >= firms["acq_date"].max() - pd.Timedelta(days=1)
    ]
    st.metric(
        label="🔥 จุดความร้อน (24h ล่าสุด)",
        value=f"{len(recent_firms):,} จุด",
        delta=f"FRP รวม {recent_firms['frp'].sum():.0f} MW",
        delta_color="off",
    )

with k4:
    temp = float(latest.get("temperature_2m", 0))
    hum  = float(latest.get("relative_humidity_2m", 0))
    st.metric(
        label="🌡️ อุณหภูมิ / ความชื้น",
        value=f"{temp:.1f} °C",
        delta=f"ความชื้น {hum:.0f}%",
        delta_color="off",
    )

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  พยากรณ์ 72h",
    "🚨  ระบบแจ้งเตือน",
    "🗺️  แผนที่จุดเสี่ยง",
    "🧠  อธิบายโมเดล (SHAP)",
])

# ── Tab 1 : Forecast ──────────────────────────────────────────────────────────
with tab1:
    st.plotly_chart(
        plot_72h_forecast(prov_data, province),
        use_container_width=True,
    )

    st.subheader("📋 สรุปพยากรณ์ราย 24 ชั่วโมง")
    daily_rows = fore72.copy()
    daily_rows["วันที่"] = daily_rows["Datetime"].dt.strftime("%d %b %Y")
    daily_tbl = (
        daily_rows.groupby("วันที่")["predicted"]
        .agg(["mean", "max", "min"])
        .rename(columns={"mean": "เฉลี่ย", "max": "สูงสุด", "min": "ต่ำสุด"})
        .reset_index()
    )
    daily_tbl["ระดับ"] = daily_tbl["สูงสุด"].apply(
        lambda x: pm25_level_info(x)["emoji"] + "  " + pm25_level_info(x)["label"]
    )
    for col in ["เฉลี่ย", "สูงสุด", "ต่ำสุด"]:
        daily_tbl[col] = daily_tbl[col].round(1).astype(str) + " µg/m³"
    st.dataframe(daily_tbl, use_container_width=True, hide_index=True)

# ── Tab 2 : Alert System ──────────────────────────────────────────────────────
with tab2:
    render_alert_section(pm25_pred72, province, pm25_trend)

    st.divider()
    st.subheader("📅 ปฏิทินความเสี่ยง (72 ชั่วโมงข้างหน้า)")

    daily_cal = fore72.copy()
    daily_cal["date"] = daily_cal["Datetime"].dt.date
    daily_max = daily_cal.groupby("date")["predicted"].max().reset_index()

    cal_cols = st.columns(len(daily_max))
    for col, (_, row) in zip(cal_cols, daily_max.iterrows()):
        info = pm25_level_info(row["predicted"])
        with col:
            st.markdown(
                f"""<div style='background:{info["color"]}1a;border:1px solid {info["color"]}44;
                         text-align:center;padding:12px 6px;border-radius:8px'>
                  <div style='font-size:26px'>{info["emoji"]}</div>
                  <div style='font-size:11px;color:#ccc'>{row["date"].strftime("%d %b")}</div>
                  <div style='font-weight:bold;color:{info["color"]};font-size:16px'>
                    {row["predicted"]:.0f}
                  </div>
                  <div style='font-size:10px;color:#aaa'>{info["label"]}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ── Tab 3 : Hotspot Priority Map ──────────────────────────────────────────────
with tab3:
    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.subheader("⚙️ ตั้งค่า")
        days_back     = st.slider("ย้อนหลัง (วัน)", 1, 90, 30)
        top_n         = st.slider("Top N จุดสำคัญ", 5, 50, 20)
        wind_override = st.number_input(
            "ทิศทางลม (องศา)", 0.0, 360.0, float(wind_deg), step=5.0,
            help="0=เหนือ · 90=ตะวันออก · 180=ใต้ · 270=ตะวันตก",
        )
        st.divider()
        st.caption(
            "🎯 **ลำดับความสำคัญ** คำนวณจาก:\n"
            "- ความรุนแรงไฟ (FRP)\n"
            "- ระยะทางถึงตัวเมือง\n"
            "- ทิศทางลม (จุดต้นลม = สำคัญกว่า)\n\n"
            "ไฮไลต์วงเหลือง = จุดที่ดับไฟแล้วลด PM2.5 ได้มากที่สุด"
        )

        # Top-5 table
        c = PROVINCE_COORDS[province]
        df_f = firms[
            firms["acq_date"] >= firms["acq_date"].max() - pd.Timedelta(days=days_back)
        ].copy()
        if len(df_f) > 0:
            dlat_f = np.radians(df_f["latitude"]  - c["lat"])
            dlon_f = np.radians(df_f["longitude"] - c["lon"])
            a_f    = (
                np.sin(dlat_f / 2) ** 2
                + np.cos(np.radians(c["lat"]))
                * np.cos(np.radians(df_f["latitude"]))
                * np.sin(dlon_f / 2) ** 2
            )
            df_f["dist_km"]  = 6371 * 2 * np.arcsin(np.sqrt(a_f.clip(0, 1)))
            bearing_f        = np.degrees(np.arctan2(df_f["longitude"] - c["lon"],
                                                      df_f["latitude"]  - c["lat"])) % 360
            angle_f          = np.abs(((bearing_f - wind_override + 180) % 360) - 180)
            wf_f             = (1 + np.cos(np.radians(angle_f))) / 2
            df_f["priority"] = (df_f["frp"].clip(0) * (0.5 + wf_f)) / (df_f["dist_km"] + 10)

            top5 = (
                df_f.nlargest(5, "priority")
                [["latitude", "longitude", "frp", "dist_km", "priority"]]
                .round(2)
                .rename(columns={
                    "latitude": "Lat", "longitude": "Lon",
                    "frp": "FRP (MW)", "dist_km": "ระยะ (กม.)",
                    "priority": "Priority",
                })
            )
            st.caption("🏆 Top 5 จุดเป้าหมาย:")
            st.dataframe(top5, use_container_width=True, hide_index=True)

    with col_map:
        recent_firms = firms[
            firms["acq_date"] >= firms["acq_date"].max() - pd.Timedelta(days=days_back)
        ]
        if len(recent_firms) == 0:
            st.info("ไม่มีข้อมูลจุดความร้อนในช่วงเวลาที่เลือก")
        else:
            st.plotly_chart(
                plot_hotspot_priority_map(recent_firms, province, wind_override, top_n),
                use_container_width=True,
            )
            st.caption(
                f"แสดง **{len(recent_firms):,}** จุดความร้อน "
                f"ย้อนหลัง {days_back} วัน จากข้อมูล NASA FIRMS VIIRS"
            )

# ── Tab 4 : SHAP Explainability ───────────────────────────────────────────────
with tab4:
    st.subheader("🧠 อธิบายการตัดสินใจของโมเดล (SHAP)")
    st.caption(
        "SHAP (SHapley Additive exPlanations) บอกว่าแต่ละตัวแปรมีผลต่อการพยากรณ์อย่างไร  "
        "**สีแดง** = ทำให้ค่าฝุ่นสูงขึ้น  |  **สีเขียว** = ทำให้ค่าฝุ่นลดลง"
    )

    X_latest = prov_data[feature_names].fillna(0)
    pm25_latest_pred = float(prov_data["predicted"].iloc[-1])

    # Insight summary sentence
    summary_html = get_shap_summary_html(model, X_latest, feature_names, pm25_latest_pred)
    if summary_html:
        st.markdown(
            f"<div style='background:#1a237e2a;border-left:4px solid #42a5f5;"
            f"padding:14px;border-radius:8px;margin-bottom:16px'>{summary_html}</div>",
            unsafe_allow_html=True,
        )

    col_wf, col_imp = st.columns(2)

    with col_wf:
        st.markdown("#### ⚡ SHAP Waterfall — การพยากรณ์ล่าสุด")
        with st.spinner("คำนวณ SHAP values..."):
            wf_fig = plot_shap_waterfall(model, X_latest, feature_names)
        if wf_fig:
            st.plotly_chart(wf_fig, use_container_width=True)
        else:
            st.warning(
                "ไม่พบ library `shap`\n\n"
                "ติดตั้งด้วยคำสั่ง: `pip install shap`"
            )

    with col_imp:
        st.markdown("#### 📊 Feature Importance รวม (XGBoost Gain)")
        st.plotly_chart(
            plot_feature_importance(model, feature_names),
            use_container_width=True,
        )

    st.divider()
    st.subheader("💡 ความหมายของ Feature สำคัญ")
    explanations = {
        "pm25_lag_24h":         "ค่าฝุ่น PM2.5 เมื่อ 24 ชั่วโมงก่อน — ตัวทำนายที่แม่นที่สุด",
        "pm25_roll_mean_24h":   "ค่าเฉลี่ย PM2.5 ใน 24h — แสดงแนวโน้มระยะสั้น",
        "pm25_roll_mean_168h":  "ค่าเฉลี่ย PM2.5 ใน 7 วัน — แสดงแนวโน้มระยะยาว",
        "hotspot_log":          "จำนวนจุดความร้อน (log scale) จาก NASA FIRMS",
        "frp_sum_log":          "Fire Radiative Power รวม — วัดความรุนแรงของไฟ",
        "hotspot_x_haze":       "จุดความร้อน × ฤดูหมอกควัน — Interaction ช่วงวิกฤต",
        "wind_dir_sin / cos":   "ทิศทางลม (cyclical encoding) — บอกว่าควันพัดมาทางไหน",
        "is_haze_season":       "อยู่ในฤดูหมอกควัน (ม.ค.–เม.ย.) หรือไม่",
        "temperature_2m":       "อุณหภูมิ — อากาศร้อนแห้งเพิ่มความเสี่ยงไฟป่า",
        "relative_humidity_2m": "ความชื้นสัมพัทธ์ — ความชื้นต่ำ = ไฟลามง่าย",
        "pm25_delta_24h":       "การเปลี่ยนแปลงค่าฝุ่นใน 24h — จับ trend ขาขึ้น/ขาลง",
    }
    rows = list(explanations.items())
    c1, c2 = st.columns(2)
    for i, (feat, desc) in enumerate(rows):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"**`{feat}`** — {desc}")

