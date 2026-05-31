"""Reusable UI components for PM2.5 Early Warning Dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap
from config import IS_HAZE_MONTHS

def clean_html(html_str: str) -> str:
    """Helper to strip all leading/trailing whitespace from each line of HTML to prevent Markdown code blocks."""
    return "\n".join(line.strip() for line in html_str.splitlines())

# ─── Constants ────────────────────────────────────────────────────────────────
# Thailand PM2.5 standards (µg/m³)
LEVELS = [
    (25,          "ดีมาก",               "#00e676", "🟢", "อากาศดีเยี่ยม ทำกิจกรรมกลางแจ้งได้"),
    (37,          "ดี",                   "#c6ff00", "🟡", "อากาศดี ทำกิจกรรมได้ตามปกติ"),
    (50,          "ปานกลาง",             "#ff9100", "🟠", "กลุ่มเสี่ยงควรลดกิจกรรมกลางแจ้ง"),
    (90,          "เริ่มมีผลต่อสุขภาพ",  "#ff1744", "🔴", "ทุกคนควรลดกิจกรรมกลางแจ้ง"),
    (float("inf"),"มีผลต่อสุขภาพ",       "#d500f9", "🟣", "หยุดทุกกิจกรรมกลางแจ้ง สวมหน้ากาก N95"),
]

PROVINCE_COORDS = {
    "Chiang Mai":   {"lat": 18.7883, "lon": 98.9853},
    "Chiang Rai":   {"lat": 19.9105, "lon": 99.8253},
    "Mae Hong Son": {"lat": 19.3003, "lon": 97.9654},
    "Lamphun":      {"lat": 18.5745, "lon": 99.0087},
    "Lampang":      {"lat": 18.2888, "lon": 99.4930},
    "Phayao":       {"lat": 19.1666, "lon": 99.9022},
    "Phrae":        {"lat": 18.1446, "lon": 100.1403},
    "Nan":          {"lat": 18.7756, "lon": 100.7730},
}

PROVINCE_LABELS = {p: i for i, p in enumerate(sorted(PROVINCE_COORDS.keys()))}

FEATURE_DISPLAY_NAMES = {
    "pm25_lag_24h":         "PM2.5 เมื่อ 24h ก่อน",
    "pm25_lag_72h":         "PM2.5 เมื่อ 72h ก่อน",
    "pm25_roll_mean_24h":   "PM2.5 เฉลี่ย 24h",
    "pm25_roll_mean_168h":  "PM2.5 เฉลี่ย 7 วัน",
    "pm25_roll_max_24h":    "PM2.5 สูงสุด 24h",
    "hotspot_log":          "จุดความร้อน (log)",
    "frp_sum_log":          "ความรุนแรงไฟรวม (FRP log)",
    "frp_mean_log":         "ความรุนแรงไฟเฉลี่ย",
    "hotspot_x_haze":       "จุดร้อน × ฤดูหมอก",
    "frp_x_haze":           "FRP × ฤดูหมอก",
    "wind_dir_sin":         "ทิศทางลม (sin)",
    "wind_dir_cos":         "ทิศทางลม (cos)",
    "wind_x_hotspot":       "ลม × จุดร้อน",
    "is_haze_season":       "ฤดูหมอกควัน",
    "temperature_2m":       "อุณหภูมิ (°C)",
    "relative_humidity_2m": "ความชื้น (%)",
    "temp_x_humidity":      "อุณหภูมิ × ความชื้น",
    "surface_pressure":     "ความกดอากาศ",
    "wind_speed_10m":       "ความเร็วลม",
    "precipitation_log":    "ฝน (log)",
    "pm25_delta_24h":       "การเปลี่ยนแปลง PM2.5 24h",
}

# ─── Utility ──────────────────────────────────────────────────────────────────

def pm25_level_info(value: float) -> dict:
    """Return AQI level metadata dict for a given PM2.5 value."""
    for threshold, label, color, emoji, advice in LEVELS:
        if value <= threshold:
            return dict(label=label, color=color, emoji=emoji, advice=advice)
    return dict(label="มีผลต่อสุขภาพ", color="#d500f9", emoji="🟣",
                advice="หยุดทุกกิจกรรมกลางแจ้ง สวมหน้ากาก N95")


def pm25_level_indicator(pm25_value):
    """Backward-compatible wrapper."""
    info = pm25_level_info(pm25_value)
    return info["label"], info["emoji"]


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_province_features(df: pd.DataFrame, province: str,
                             province_target_enc: dict) -> pd.DataFrame:
    """
    Reconstruct the 69-feature matrix from the merged CSV for a single province.

    Required input columns:
        Datetime, PM25, temperature_2m, relative_humidity_2m, precipitation,
        surface_pressure, wind_speed_10m, wind_direction_10m,
        hotspot_count, frp_sum, frp_mean
    """
    d = df.sort_values("Datetime").copy().reset_index(drop=True)
    dt = d["Datetime"]

    # Log transforms
    d["precipitation_log"] = np.log1p(d["precipitation"])
    d["hotspot_log"]       = np.log1p(d["hotspot_count"])
    d["frp_sum_log"]       = np.log1p(d["frp_sum"])
    d["frp_mean_log"]      = np.log1p(d["frp_mean"])

    # Wind — cyclical encoding
    wr = np.radians(d["wind_direction_10m"])
    d["wind_dir_sin"] = np.sin(wr)
    d["wind_dir_cos"] = np.cos(wr)

    # Time — cyclical encoding
    d["hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour  / 24)
    d["hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour  / 24)
    d["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    d["day_sin"]   = np.sin(2 * np.pi * dt.dt.day   / 31)
    d["day_cos"]   = np.cos(2 * np.pi * dt.dt.day   / 31)
    d["is_haze_season"] = dt.dt.month.isin([1, 2, 3, 4]).astype(int)
    d["year"]           = dt.dt.year

    # Province encoding
    d["province_label"]      = PROVINCE_LABELS.get(province, 0)
    d["province_target_enc"] = province_target_enc.get(province, d["PM25"].mean())

    # Interaction features
    d["temp_x_humidity"] = d["temperature_2m"] * d["relative_humidity_2m"]
    d["hotspot_x_haze"]  = d["hotspot_log"] * d["is_haze_season"]
    d["frp_x_haze"]      = d["frp_sum_log"] * d["is_haze_season"]
    d["wind_x_hotspot"]  = d["wind_speed_10m"] * d["hotspot_log"]

    # PM2.5 lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        d[f"pm25_lag_{lag}h"] = d["PM25"].shift(lag)

    # Hotspot / FRP lag features
    for lag in [24, 48, 72]:
        d[f"hotspot_lag_{lag}h"]     = d["hotspot_count"].shift(lag)
        d[f"frp_sum_lag_{lag}h"]     = d["frp_sum"].shift(lag)
        d[f"hotspot_log_lag_{lag}h"] = d["hotspot_log"].shift(lag)

    # PM2.5 rolling features
    for w in [3, 6, 12, 24, 48, 168]:
        d[f"pm25_roll_mean_{w}h"] = d["PM25"].rolling(w).mean()
        d[f"pm25_roll_std_{w}h"]  = d["PM25"].rolling(w).std()
        d[f"pm25_roll_max_{w}h"]  = d["PM25"].rolling(w).max()

    # Hotspot / FRP rolling features
    for w in [24, 48, 168]:
        d[f"hotspot_roll_sum_{w}h"] = d["hotspot_count"].rolling(w).sum()
        d[f"frp_roll_sum_{w}h"]     = d["frp_sum"].rolling(w).sum()

    # Delta features
    d["pm25_delta_1h"]      = d["PM25"].diff(1)
    d["pm25_delta_24h"]     = d["PM25"].diff(24)
    d["humidity_delta_1h"]  = d["relative_humidity_2m"].diff(1)
    d["humidity_delta_24h"] = d["relative_humidity_2m"].diff(24)

    return d.dropna(subset=["pm25_lag_72h"]).reset_index(drop=True)


# ─── Section 1: 7-Day Forecast ──────────────────────────────────────────────

def plot_7day_forecast(prov_data: pd.DataFrame, province: str, now: pd.Timestamp = None) -> go.Figure:
    """
    Ultra-polished 7-Day Forecast Chart:
      - Smooth spline curves
      - Clean annotations and 'Now' indicator
      - Subtle confidence bands and grid
      - Removed range slider for a cleaner look
    """
    if now is None:
        now = pd.Timestamp.now().floor('h')
        
    d = prov_data.sort_values("Datetime").tail(14 * 24).copy()
    hist = d[d["is_predicted"] == False]
    fore = d[d["is_predicted"] == True]
    if not hist.empty and not fore.empty:
        fore = pd.concat([hist.iloc[[-1]], fore], ignore_index=True)

    fig = go.Figure()

    # 1. Subtle Background Zones for Danger Levels
    fig.add_hrect(y0=37.5, y1=75, fillcolor="rgba(255, 145, 0, 0.03)", line_width=0, layer="below")
    fig.add_hrect(y0=75, y1=500, fillcolor="rgba(255, 23, 68, 0.03)", line_width=0, layer="below")

    # 2. Historical Actual (Smooth Line - Premium Theme Blue)
    fig.add_trace(go.Scatter(
        x=hist["Datetime"], y=hist["PM25"],
        name="อดีต (Actual)",
        line=dict(color="#0f62fe", width=3, shape='spline', smoothing=0.8),
        mode="lines",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>PM2.5 จริง: %{y:.1f} µg/m³<extra></extra>"
    ))

    if not fore.empty:
        # 3. Confidence Band (Smooth and highly transparent)
        upper = fore["predicted"] * 1.15
        lower = fore["predicted"] * 0.85
        fig.add_trace(go.Scatter(
            x=pd.concat([fore["Datetime"], fore["Datetime"].iloc[::-1]]),
            y=pd.concat([upper, lower.iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(15, 98, 254, 0.05)',
            line=dict(color='rgba(255,255,255,0)', shape='spline', smoothing=0.8),
            hoverinfo="skip",
            showlegend=True,
            name="ช่วงความไม่แน่นอน"
        ))

        # 4. Forecast Line (Smooth Area - Vibrant Prediction Coral/Orange)
        fig.add_trace(go.Scatter(
            x=fore["Datetime"], y=fore["predicted"],
            name="พยากรณ์ (Forecast)",
            line=dict(color="#ff7043", width=3.5, shape='spline', smoothing=0.8),
            fill='tozeroy',
            fillcolor='rgba(255, 112, 67, 0.05)',
            mode="lines",
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>พยากรณ์: %{y:.1f} µg/m³<br><extra></extra>"
        ))

        # 5. Highlight Peak Point with a nice annotation
        peak_idx = fore["predicted"].idxmax()
        peak_row = fore.loc[peak_idx]
        fig.add_annotation(
            x=peak_row["Datetime"],
            y=peak_row["predicted"],
            text=f"สูงสุด {peak_row['predicted']:.0f} µg/m³",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#0f62fe",
            ax=0,
            ay=-40,
            font=dict(color="#0f62fe", size=12, family="Prompt, sans-serif"),
            bgcolor="rgba(15, 98, 254, 0.08)",
            bordercolor="#0f62fe",
            borderwidth=1,
            borderpad=4,
            opacity=0.95
        )

    # 6. 'Now' Vertical Line (Clean Slate Gray)
    fig.add_shape(
        type="line",
        x0=now, x1=now, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#64748b", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=now, y=1,
        xref="x", yref="paper",
        text=" ปัจจุบัน",
        showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color="#64748b", size=11, family="Prompt, sans-serif")
    )

    # 7. Threshold Lines
    fig.add_hline(y=37.5, line_dash="dot", line_color="#eab308", line_width=1.5)
    fig.add_annotation(x=0.01, y=37.5, xref="paper", yref="y", text="เริ่มมีผลกระทบ (37.5)", showarrow=False, font=dict(color="#eab308", size=10, family="Prompt, sans-serif"), yanchor="bottom")
    
    fig.add_hline(y=75, line_dash="dot", line_color="#ef4444", line_width=1.5)
    fig.add_annotation(x=0.01, y=75, xref="paper", yref="y", text="อันตราย (75)", showarrow=False, font=dict(color="#ef4444", size=10, family="Prompt, sans-serif"), yanchor="bottom")

    # 8. Clean Layout
    y_max = max(150, float(prov_data["PM25"].max()) * 1.1, float(fore["predicted"].max() if not fore.empty else 0) * 1.1)
    
    fig.update_layout(
        title=dict(
            text=f"แนวโน้มฝุ่น PM2.5 ล่วงหน้า 7 วัน — <b>{province}</b>",
            font=dict(size=16, family="Prompt, 'Plus Jakarta Sans', sans-serif", color="#1e293b")
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            griddash="solid",
            showline=False,
            zeroline=False,
            tickfont=dict(family="'Plus Jakarta Sans', 'Prompt', sans-serif", color="#64748b", size=11)
        ),
        yaxis=dict(
            title=dict(
                text="ปริมาณฝุ่น PM2.5 (µg/m³)",
                font=dict(family="Prompt, sans-serif", color="#64748b", size=12)
            ),
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(family="'Plus Jakarta Sans', sans-serif", color="#64748b", size=11),
            range=[0, y_max]
        ),
        height=450,
        margin=dict(l=40, r=20, t=60, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, family="Prompt, sans-serif", color="#475569")
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b")
    )

    return fig


# ─── Section 2: Alert System ──────────────────────────────────────────────────

def render_alert_section(pm25_max: float, province: str, pm25_trend: float = 0.0):
    """Overall alert banner + 3 risk-group cards + 7-day calendar."""
    info       = pm25_level_info(pm25_max)
    trend_icon = "📈" if pm25_trend > 2 else "📉" if pm25_trend < -2 else "➡️"
    trend_txt  = (
        f"{trend_icon} แนวโน้ม "
        f"{'สูงขึ้น' if pm25_trend > 0 else 'ต่ำลง'} "
        f"{abs(pm25_trend):.1f} µg/m³ เทียบกับสัปดาห์ก่อน"
    )

    st.markdown(
        clean_html(f"""
        <div style="background: rgba(255, 255, 255, 0.85);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                    border-left: 6px solid {info['color']};
                    box-shadow: 0 10px 25px -5px rgba(148, 163, 184, 0.08), 0 4px 12px -2px rgba(148, 163, 184, 0.03);
                    padding: 24px 28px;
                    border-radius: 16px;
                    margin-bottom: 28px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    backdrop-filter: blur(10px);
                    font-family: 'Prompt', sans-serif;">
          <div>
            <h2 style="margin: 0; color: {info['color']}; font-family: 'Prompt', sans-serif; font-size: 20px; font-weight: 800; display: flex; align-items: center; gap: 8px;">
               {info['emoji']} {info['label']}
            </h2>
            <p style="margin: 10px 0 0; color: #475569; font-size: 14px; font-weight: 500; line-height: 1.5;">
              ความเสี่ยงสูงสุดในอีก 7 วันข้างหน้า: 
              <b style="color: {info['color']}; font-size: 16px; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; background: {info['color']}12; padding: 2px 8px; border-radius: 6px;">{pm25_max:.1f} µg/m³</b>
              &nbsp;|&nbsp; <span style="font-weight: 600; color: #334155; font-size: 13px;">{trend_txt}</span>
            </p>
            <div style="margin: 12px 0 0; color: #64748b; font-size: 13px; display: flex; align-items: center; gap: 10px;">
              <span style="background: {info['color']}15; padding: 3px 10px; border-radius: 6px; font-weight: 700; color: {info['color']}; font-size: 11px; text-transform: uppercase;">คำแนะนำ</span>
              <span style="color: #475569; font-weight: 500;">{info['advice']}</span>
            </div>
          </div>
          <div style="font-size: 52px; background: {info['color']}10; width: 84px; height: 84px; display: flex; align-items: center; justify-content: center; border-radius: 50%; opacity: 0.9; margin-left: 20px;">
            {info['emoji']}
          </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    groups = [
        {
            "title":  "🏥 โรงพยาบาล & คลินิก",
            "alert":  pm25_max > 50,
            "actions": [
                "เตรียมรับผู้ป่วยโรคระบบทางเดินหายใจเพิ่ม",
                "เพิ่มสต็อกยาพ่น / หน้ากาก N95",
                "แจ้งแผนกฉุกเฉินเตรียมพร้อม",
            ],
        },
        {
            "title":  "🏫 โรงเรียน & มหาวิทยาลัย",
            "alert":  pm25_max > 37,
            "actions": [
                "ยกเลิก / เลื่อนกิจกรรมกีฬากลางแจ้ง",
                "แจ้งผู้ปกครองและนักเรียน",
                "ตรวจสอบระบบกรองอากาศ HEPA",
            ],
        },
        {
            "title":  "👴 กลุ่มเสี่ยง (เด็ก / สูงอายุ / ผู้ป่วย)",
            "alert":  pm25_max > 37,
            "actions": [
                "อยู่ในอาคาร ปิดหน้าต่างและประตู",
                "สวมหน้ากาก N95 หากออกนอกบ้าน",
                "หลีกเลี่ยงออกกำลังกายกลางแจ้ง",
            ],
        },
    ]

    cols = st.columns(3)
    for col, g in zip(cols, groups):
        border = "#ef4444" if g["alert"] else "#10b981"
        status = "⚠️ แจ้งเตือนระดับสูง" if g["alert"] else "✅ ปกติและปลอดภัย"
        li = "".join(
            f"<li style='margin: 8px 0; color: #475569; font-size: 13px; line-height: 1.5; list-style-type: none; display: flex; align-items: flex-start; gap: 8px; font-weight: 500;'>"
            f"<span style='color: {border}; font-size: 10px; margin-top: 1px;'>◆</span><span>{a}</span></li>"
            for a in g["actions"]
        )
        with col:
            st.markdown(
                clean_html(f"""
                <div class="premium-card" style="border-top: 4px solid {border}; min-height: 200px; display: flex; flex-direction: column; justify-content: flex-start;">
                  <div style="margin-bottom: 12px;">
                    <span style="background: {border}12; color: {border}; font-size: 10px; font-weight: 800;
                                 padding: 4px 12px; border-radius: 9999px; display: inline-block; letter-spacing: 0.5px;">{status}</span>
                  </div>
                  <b style="font-size: 15px; color: #1e293b; font-family: 'Prompt', sans-serif; font-weight: 700; margin-bottom: 6px; display: block;">{g["title"]}</b>
                  <ul style="padding: 0; margin: 6px 0 0; font-family: 'Prompt', sans-serif;">
                    {li}
                  </ul>
                </div>
                """),
                unsafe_allow_html=True,
            )


# ─── Section 3: Hotspot Priority Map ─────────────────────────────────────────

def plot_hotspot_priority_map(
    firms_df: pd.DataFrame,
    province: str,
    wind_deg: float = 0.0,
    top_n: int = 20,
) -> go.Figure:
    """
    Hotspot map with priority score:
        priority = frp × (0.5 + wind_alignment) / (distance_km + 10)

    High priority = intense fire + close to province + upwind position.
    Top-N hotspots are highlighted with yellow rings.
    """
    coords = PROVINCE_COORDS[province]
    cx, cy = coords["lat"], coords["lon"]

    df = firms_df.copy()
    
    # 4.5 Cross-border Hotspot identification (Simple BBox fallback)
    def identify_country(lat, lon):
        if 5.6 <= lat <= 20.5 and 97.3 <= lon <= 105.6:
            return "Thailand"
        return "Cross-border"
    
    df["country"] = df.apply(lambda r: identify_country(r["latitude"], r["longitude"]), axis=1)

    # Haversine distance to province centroid
    dlat = np.radians(df["latitude"]  - cx)
    dlon = np.radians(df["longitude"] - cy)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(cx))
        * np.cos(np.radians(df["latitude"]))
        * np.sin(dlon / 2) ** 2
    )
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))

    # Wind alignment: hotspots upwind of province score higher
    bearing    = np.degrees(np.arctan2(df["longitude"] - cy, df["latitude"] - cx)) % 360
    angle_diff = np.abs(((bearing - wind_deg + 180) % 360) - 180)
    wind_factor = (1 + np.cos(np.radians(angle_diff))) / 2  # 0=downwind → 1=upwind

    df["priority_score"] = (df["frp"].clip(0) * (0.5 + wind_factor)) / (df["distance_km"] + 10)

    fig = px.scatter_mapbox(
        df,
        lat="latitude", lon="longitude",
        color="priority_score",
        size=df["frp"].clip(1, 400).values,
        color_continuous_scale="YlOrRd",
        size_max=18,
        zoom=6,
        center={"lat": cx, "lon": cy},
        mapbox_style="carto-positron",
        hover_data={
            "latitude": False, "longitude": False,
            "priority_score": ":.3f", "frp": ":.1f", "distance_km": ":.0f",
            "country": True
        },
        labels={"priority_score": "คะแนนความสำคัญ", "frp": "FRP (MW)",
                "distance_km": "ระยะ (กม.)", "country": "ประเทศ"},
        title=f"🗺️ แผนที่จุดความร้อน & ลำดับความสำคัญ — {province}",
    )

    # Yellow rings around Top-N priority hotspots
    top = df.nlargest(top_n, "priority_score")
    fig.add_trace(go.Scattermapbox(
        lat=top["latitude"], lon=top["longitude"],
        mode="markers",
        marker=dict(size=22, color="rgba(255,160,0,0.3)"),
        name=f"🎯 Top {top_n} เป้าหมายดับไฟ",
        hoverinfo="skip",
    ))

    # Province centroid marker
    fig.add_trace(go.Scattermapbox(
        lat=[cx], lon=[cy],
        mode="markers+text",
        marker=dict(size=14, color="#0f62fe"),
        text=[f"📍 {province}"],
        textposition="bottom right",
        textfont=dict(color="#1e293b", size=13, family="Prompt, sans-serif"),
        name=province,
    ))

    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Prompt, sans-serif"),
        legend=dict(orientation="h", y=-0.06, font=dict(family="Prompt, sans-serif")),
        coloraxis_colorbar=dict(
            title=dict(
                text="Priority",
                font=dict(family="Prompt, sans-serif")
            ),
            thickness=12
        ),
    )
    return fig


# ─── Section 4: SHAP Explainability ──────────────────────────────────────────

def _compute_shap(model, X_row: pd.DataFrame):
    """Compute SHAP values for a single row. Returns (shap_values, base_val) or None."""
    try:
        import shap
    except ImportError:
        return None, None
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    return shap_values[0], float(explainer.expected_value)


def plot_shap_waterfall(model, X_latest: pd.DataFrame,
                        feature_names: list, precomputed_shap: pd.DataFrame = None, 
                        province: str = None) -> go.Figure | None:
    """
    SHAP waterfall bar chart for the latest prediction.
    Red bars = features pushing PM2.5 higher.
    Green bars = features pushing PM2.5 lower.
    """
    sv = None
    base_val = 0.0
    final_pred = 0.0
    feat_df = pd.DataFrame()

    # 1. Try to use precomputed data
    if precomputed_shap is not None and province is not None:
        prov_shap = precomputed_shap[precomputed_shap['Province'] == province]
        if not prov_shap.empty:
            base_val = float(prov_shap['base_value'].iloc[0])
            final_pred = float(prov_shap['predicted_pm25'].iloc[0])
            feat_df = (
                prov_shap.rename(columns={'feature_name': 'feature', 'shap_value': 'shap', 'feature_value': 'value'})
                .assign(abs_shap=lambda d: d["shap"].abs())
                .nlargest(15, "abs_shap")
            )
            sv = feat_df['shap'].values # Just to trigger next steps

    # 2. Fallback to live computation
    if sv is None:
        sv, base_val = _compute_shap(model, X_latest.iloc[[-1]])
        if sv is None:
            return None
        
        final_pred = base_val + float(sv.sum())
        feat_df = (
            pd.DataFrame({
                "feature": feature_names,
                "shap":    sv,
                "value":   X_latest.iloc[-1].values,
            })
            .assign(abs_shap=lambda d: d["shap"].abs())
            .nlargest(15, "abs_shap")
        )

    colors = ["#ef5350" if v > 0 else "#66bb6a" for v in feat_df["shap"]]
    labels = [
        FEATURE_DISPLAY_NAMES.get(r["feature"], r["feature"])
        + f" ({r['value']:.1f})"
        for _, r in feat_df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=feat_df["shap"], y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in feat_df["shap"]],
        textposition="outside",
        hovertemplate="%{y}<br>SHAP: %{x:+.2f} µg/m³<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1.5)

    fig.update_layout(
        title=dict(
            text=(
                f"SHAP Waterfall — โมเดลพยากรณ์ <b>{final_pred:.1f} µg/m³</b>"
                f"<br><sup>Base = {base_val:.1f} | แดง = เพิ่มฝุ่น | เขียว = ลดฝุ่น</sup>"
            ),
            font=dict(size=14, family="Prompt, sans-serif", color="#1e293b"),
        ),
        xaxis=dict(
            title=dict(
                text="SHAP Value (µg/m³)",
                font=dict(family="Prompt, sans-serif", color="#64748b", size=12)
            ),
            tickfont=dict(family="Inter, sans-serif", color="#64748b")
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Prompt, sans-serif", color="#1e293b", size=11)
        ),
        height=480,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b"),
        margin=dict(l=240, r=60),
    )
    return fig


def plot_feature_importance(model, feature_names: list) -> go.Figure:
    """Global feature importance bar chart (XGBoost Gain)."""
    imp = (
        pd.DataFrame({"feature": feature_names,
                      "importance": model.feature_importances_})
        .assign(display=lambda d: d["feature"].map(
            lambda x: FEATURE_DISPLAY_NAMES.get(x, x)))
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    fig = px.bar(
        imp, x="importance", y="display", orientation="h",
        color="importance", color_continuous_scale=["#93c5fd", "#0f62fe"],
        labels={"importance": "Importance (Gain)", "display": "Feature"},
        title="📊 Global Feature Importance — Top 20 (XGBoost Gain)",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Prompt, sans-serif"), height=520,
        title=dict(
            font=dict(size=14, color="#1e293b", family="Prompt, sans-serif")
        ),
        xaxis=dict(tickfont=dict(family="Inter, sans-serif", color="#64748b")),
        yaxis=dict(tickfont=dict(family="Prompt, sans-serif", color="#1e293b", size=11)),
        showlegend=False, coloraxis_showscale=False,
    )
    return fig


def get_shap_summary_html(model, X_latest: pd.DataFrame,
                          feature_names: list, pm25_pred: float,
                          precomputed_shap: pd.DataFrame = None, 
                          province: str = None) -> str:
    """
    Return an HTML sentence explaining the top driving factors.
    e.g. "โมเดลคาดว่าฝุ่นจะสูง 82.3 µg/m³ เพราะ: จุดความร้อน 42%, ค่าฝุ่นเมื่อวาน 31%"
    """
    feat_df = None

    # 1. Try precomputed
    if precomputed_shap is not None and province is not None:
        prov_shap = precomputed_shap[precomputed_shap['Province'] == province]
        if not prov_shap.empty:
            feat_df = prov_shap.rename(columns={'feature_name': 'feature', 'shap_value': 'shap'})

    # 2. Fallback live
    if feat_df is None:
        sv, _ = _compute_shap(model, X_latest.iloc[[-1]])
        if sv is None:
            return ""
        feat_df = pd.DataFrame({"feature": feature_names, "shap": sv})

    total_abs = np.abs(feat_df["shap"]).sum() + 1e-9

    # Top positive contributors (things making PM2.5 go UP)
    pos = feat_df[feat_df["shap"] > 0].nlargest(3, "shap")
    parts = []
    for _, row in pos.iterrows():
        pct  = abs(row["shap"]) / total_abs * 100
        name = FEATURE_DISPLAY_NAMES.get(row["feature"], row["feature"])
        parts.append(f"<b>{name}</b> {pct:.0f}%")

    if not parts:
        return ""
    return (
        f"💬 โมเดลคาดว่าฝุ่นจะ <b style='color:#ff7043'>{pm25_pred:.1f} µg/m³</b> "
        f"เพราะ: " + " &nbsp;·&nbsp; ".join(parts)
    )


def render_province_overview(all_data: pd.DataFrame, now: pd.Timestamp):
    """
    Render 8-province ranking cards based on max predicted PM2.5 in next 24h.
    """
    # 1. Prepare data
    forecast_24h = all_data[
        (all_data["Datetime"] > now) & 
        (all_data["Datetime"] <= now + pd.Timedelta(hours=24))
    ]
    
    current_vals = all_data[all_data["Datetime"] <= now].groupby("Province")["PM25"].last().to_dict()
    
    ranking = (
        forecast_24h.groupby("Province")["predicted"]
        .max()
        .reset_index()
        .sort_values("predicted", ascending=False)
    )
    
    # 2. Render Grid
    st.subheader("📊 สรุปสถานการณ์ 8 จังหวัดภาคเหนือ (พยากรณ์ 24h)")
    
    cols = st.columns(4)
    for i, (_, row) in enumerate(ranking.iterrows()):
        prov = row["Province"]
        pred_max = row["predicted"]
        curr_val = current_vals.get(prov, 0)
        info = pm25_level_info(pred_max)
        
        with cols[i % 4]:
            st.markdown(
                clean_html(f"""
                <div class="premium-card" style="border-top: 4px solid {info['color']}; text-align: center; margin-bottom: 12px; min-height: 200px; padding: 20px 16px; backdrop-filter: blur(10px); background: rgba(255,255,255,0.85);">
                  <h4 style="margin: 0 0 4px; color: #1e293b; font-size: 16px; font-weight: 750; font-family: 'Prompt', sans-serif;">📍 {prov}</h4>
                  <div style="font-size: 30px; margin: 8px 0; line-height: 1;">{info["emoji"]}</div>
                  <div style="font-size: 12px; color: #64748b; margin-bottom: 6px; font-family: 'Prompt', sans-serif;">
                    ปัจจุบัน: <b style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #334155;">{curr_val:.1f}</b> µg/m³
                  </div>
                  <div style="font-size: 13px; color: #475569; font-weight: 600; margin-bottom: 8px; font-family: 'Prompt', sans-serif;">
                    พยากรณ์สูงสุด: <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: {info['color']}; font-size: 15px;">{pred_max:.1f}</span>
                  </div>
                  <div style="font-size: 10px; color: {info['color']}; font-weight: 800; background: {info['color']}12; padding: 3px 10px; border-radius: 9999px; display: inline-block; font-family: 'Prompt', sans-serif; border: 1px solid {info['color']}20;">
                    {info["label"]}
                  </div>
                </div>
                """),
                unsafe_allow_html=True,
            )
            
            # Select province button styled cleanly under the card
            if st.button(
                f"เลือก {prov}",
                key=f"btn_{prov}",
                use_container_width=True,
                help=f"ดูรายละเอียดจังหวัด {prov}"
            ):
                st.session_state["selected_province"] = prov
                st.rerun()