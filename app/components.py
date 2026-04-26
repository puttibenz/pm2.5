"""Reusable UI components for PM2.5 Early Warning Dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

def plot_7day_forecast(prov_data: pd.DataFrame, province: str) -> go.Figure:
    """
    Ultra-polished 7-Day Forecast Chart:
      - Smooth spline curves
      - Clean annotations and 'Now' indicator
      - Subtle confidence bands and grid
      - Removed range slider for a cleaner look
    """
    d = prov_data.sort_values("Datetime").tail(14 * 24).copy()
    now = pd.Timestamp.now().floor('h')
    hist = d[d["Datetime"] <= now]
    fore = d[d["Datetime"] >= now] # Include 'now' to connect the lines

    fig = go.Figure()

    # 1. Subtle Background Zones for Danger Levels
    fig.add_hrect(y0=37.5, y1=75, fillcolor="rgba(255, 145, 0, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=75, y1=500, fillcolor="rgba(255, 23, 68, 0.05)", line_width=0, layer="below")

    # 2. Historical Actual (Smooth Line)
    fig.add_trace(go.Scatter(
        x=hist["Datetime"], y=hist["PM25"],
        name="อดีต (Actual)",
        line=dict(color="#29b6f6", width=2.5, shape='spline', smoothing=0.8),
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
            fillcolor='rgba(255, 167, 38, 0.1)',
            line=dict(color='rgba(255,255,255,0)', shape='spline', smoothing=0.8),
            hoverinfo="skip",
            showlegend=True,
            name="ช่วงความไม่แน่นอน"
        ))

        # 4. Forecast Line (Smooth Area)
        fig.add_trace(go.Scatter(
            x=fore["Datetime"], y=fore["predicted"],
            name="พยากรณ์ (Forecast)",
            line=dict(color="#ff7043", width=3.5, shape='spline', smoothing=0.8),
            fill='tozeroy',
            fillcolor='rgba(255, 112, 67, 0.08)',
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
            arrowcolor="#d500f9",
            ax=0,
            ay=-40,
            font=dict(color="#d500f9", size=12),
            bgcolor="rgba(213, 0, 249, 0.1)",
            bordercolor="#d500f9",
            borderwidth=1,
            borderpad=4,
            opacity=0.9
        )

    # 6. 'Now' Vertical Line (Using Shapes and Annotations directly to avoid library version conflicts)
    fig.add_shape(
        type="line",
        x0=now, x1=now, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#90a4ae", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=now, y=1,
        xref="x", yref="paper",
        text=" ปัจจุบัน",
        showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color="#90a4ae", size=11)
    )

    # 7. Threshold Lines
    fig.add_hline(y=37.5, line_dash="dot", line_color="#ffd600", line_width=1.5)
    fig.add_annotation(x=0.01, y=37.5, xref="paper", yref="y", text="เริ่มมีผลกระทบ (37.5)", showarrow=False, font=dict(color="#ffd600", size=10), yanchor="bottom")
    
    fig.add_hline(y=75, line_dash="dot", line_color="#ff1744", line_width=1.5)
    fig.add_annotation(x=0.01, y=75, xref="paper", yref="y", text="อันตราย (75)", showarrow=False, font=dict(color="#ff1744", size=10), yanchor="bottom")

    # 8. Clean Layout
    y_max = max(150, float(prov_data["PM25"].max()) * 1.1, float(fore["predicted"].max() if not fore.empty else 0) * 1.1)
    
    fig.update_layout(
        title=dict(
            text=f"แนวโน้มฝุ่น PM2.5 ล่วงหน้า 7 วัน — <b>{province}</b>",
            font_size=18
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            gridwidth=1,
            griddash="dot",
            showline=False,
            zeroline=False
        ),
        yaxis=dict(
            title="ปริมาณฝุ่น PM2.5 (µg/m³)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            gridwidth=1,
            showline=False,
            zeroline=False,
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
            font=dict(size=12)
        ),
        plot_bgcolor="rgba(14, 17, 23, 1)",
        paper_bgcolor="rgba(14, 17, 23, 1)",
        font=dict(color="#cfd8dc")
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
        f"""
        <div style='background:{info["color"]}1a;border-left:6px solid {info["color"]};
                    padding:18px 22px;border-radius:10px;margin-bottom:20px'>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
              <h2 style='margin:0;color:{info["color"]}'>{info["emoji"]} {info["label"]}</h2>
              <p style='margin:6px 0 0;color:#ccc;font-size:15px'>
                PM2.5 สูงสุดพยากรณ์ 7 วันอันตราย:
                <b style='color:{info["color"]}'>{pm25_max:.1f} µg/m³</b>
                &nbsp;|&nbsp; {trend_txt}
              </p>
              <p style='margin:6px 0 0;color:#aaa;font-size:13px'>
                💡 {info["advice"]}
              </p>
            </div>
            <div style='font-size:56px;opacity:0.85'>{info["emoji"]}</div>
          </div>
        </div>
        """,
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
        border = "#ff1744" if g["alert"] else "#00e676"
        status = "⚠️ แจ้งเตือน" if g["alert"] else "✅ ปกติ"
        li = "".join(
            f"<li style='margin:3px 0;color:#bbb;font-size:12px'>{a}</li>"
            for a in g["actions"]
        )
        with col:
            st.markdown(
                f"""<div style='background:{border}12;border:1px solid {border}44;
                         padding:16px;border-radius:10px;min-height:175px'>
                  <span style='color:{border};font-size:12px;font-weight:bold'>{status}</span><br>
                  <b style='font-size:14px'>{g["title"]}</b>
                  <ul style='padding-left:16px;margin:8px 0 0'>{li}</ul>
                </div>""",
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
        mapbox_style="carto-darkmatter",
        hover_data={
            "latitude": False, "longitude": False,
            "priority_score": ":.3f", "frp": ":.1f", "distance_km": ":.0f",
        },
        labels={"priority_score": "คะแนนความสำคัญ", "frp": "FRP (MW)",
                "distance_km": "ระยะ (กม.)"},
        title=f"🗺️ แผนที่จุดความร้อน & ลำดับความสำคัญ — {province}",
    )

    # Yellow rings around Top-N priority hotspots
    top = df.nlargest(top_n, "priority_score")
    fig.add_trace(go.Scattermapbox(
        lat=top["latitude"], lon=top["longitude"],
        mode="markers",
        marker=dict(size=22, color="rgba(255,235,59,0.28)"),
        name=f"🎯 Top {top_n} เป้าหมายดับไฟ",
        hoverinfo="skip",
    ))

    # Province centroid marker
    fig.add_trace(go.Scattermapbox(
        lat=[cx], lon=[cy],
        mode="markers+text",
        marker=dict(size=14, color="#42a5f5"),
        text=[f"📍 {province}"],
        textposition="bottom right",
        textfont=dict(color="#fff", size=13),
        name=province,
    ))

    fig.update_layout(
        height=540,
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", y=-0.06),
        coloraxis_colorbar=dict(title="Priority", thickness=12),
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
                        feature_names: list) -> go.Figure | None:
    """
    SHAP waterfall bar chart for the latest prediction.
    Red bars = features pushing PM2.5 higher.
    Green bars = features pushing PM2.5 lower.
    """
    sv, base_val = _compute_shap(model, X_latest.iloc[[-1]])
    if sv is None:
        return None

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
    fig.add_vline(x=0, line_color="#607d8b", line_width=1.5)

    final_pred = base_val + float(sv.sum())
    fig.update_layout(
        title=dict(
            text=(
                f"SHAP Waterfall — โมเดลพยากรณ์ <b>{final_pred:.1f} µg/m³</b>"
                f"<br><sup>Base = {base_val:.1f} | แดง = เพิ่มฝุ่น | เขียว = ลดฝุ่น</sup>"
            ),
            font_size=14,
        ),
        xaxis_title="SHAP Value (µg/m³)",
        height=480,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
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
        color="importance", color_continuous_scale="plasma",
        labels={"importance": "Importance (Gain)", "display": "Feature"},
        title="📊 Global Feature Importance — Top 20 (XGBoost Gain)",
    )
    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), height=520,
        showlegend=False, coloraxis_showscale=False,
    )
    return fig


def get_shap_summary_html(model, X_latest: pd.DataFrame,
                          feature_names: list, pm25_pred: float) -> str:
    """
    Return an HTML sentence explaining the top driving factors.
    e.g. "โมเดลคาดว่าฝุ่นจะสูง 82.3 µg/m³ เพราะ: จุดความร้อน 42%, ค่าฝุ่นเมื่อวาน 31%"
    """
    sv, _ = _compute_shap(model, X_latest.iloc[[-1]])
    if sv is None:
        return ""

    feat_df = pd.DataFrame({"feature": feature_names, "shap": sv})
    total_abs = np.abs(sv).sum() + 1e-9

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
