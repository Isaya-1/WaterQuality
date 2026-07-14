import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
import os
from datetime import datetime

# =====================================================================
#  PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
#  CUSTOM CSS FOR DARK THEME
# =====================================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .metric-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 48px rgba(0,0,0,0.5); }
    .metric-label { color: #a0aec0; font-size: 14px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #ffffff; font-size: 28px; font-weight: 700; margin: 5px 0; }
    .metric-unit { color: #718096; font-size: 14px; font-weight: 400; }
    .safe-banner {
        background: linear-gradient(135deg, #00b894, #00cec9);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,206,201,0.3);
    }
    .unsafe-banner {
        background: linear-gradient(135deg, #e17055, #d63031);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(214,48,49,0.3);
    }
    .reason-text { color: #fdcb6e; font-weight: 400; font-size: 14px; }
    .section-title { color: #ffffff; font-size: 20px; font-weight: 600; margin-bottom: 20px; border-bottom: 2px solid rgba(255,255,255,0.1); padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
#  THINGSPEAK CONFIGURATION
# =====================================================================
CHANNEL_ID = "3389783"
READ_API_KEY = "04JJ8T8BA0TIIBQ0"
THINGSPEAK_URL = (
    f"https://api.thingspeak.com/channels/"
    f"{CHANNEL_ID}/feeds.json"
    f"?api_key={READ_API_KEY}&results=200"
)

# =====================================================================
#  SAFETY THRESHOLDS
# =====================================================================
PH_MIN, PH_MAX = 6.5, 8.5
TDS_MAX = 500
TURB_MAX = 5.0
TEMP_MIN, TEMP_MAX = 15, 35

# =====================================================================
#  FETCH DATA FROM THINGSPEAK
# =====================================================================
@st.cache_data(ttl=15)
def load_data():
    try:
        response = requests.get(THINGSPEAK_URL, timeout=15)
        if response.status_code != 200:
            st.error(f"ThingSpeak returned HTTP {response.status_code}")
            st.text(response.text)
            return pd.DataFrame()
        data = response.json()
        if "feeds" not in data:
            st.error("ThingSpeak response does not contain 'feeds'")
            st.json(data)
            return pd.DataFrame()
        feeds = data["feeds"]
        if not feeds:
            st.warning("No data found in ThingSpeak.")
            return pd.DataFrame()
        df = pd.DataFrame(feeds)
        # CORRECT MAPPING: field1=Temperature, field2=pH, field3=TDS, field4=Turbidity
        df.rename(columns={
            "field1": "Temperature",
            "field2": "pH",
            "field3": "TDS",
            "field4": "Turbidity"
        }, inplace=True)
        df["created_at"] = pd.to_datetime(df["created_at"])
        for col in ["Temperature", "pH", "TDS", "Turbidity"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# =====================================================================
#  LOAD ML MODELS (with fallback for xgboost_model.pkl)
# =====================================================================
@st.cache_resource
def load_ml_models():
    try:
        import joblib
        # Check if files exist
        files_present = {}
        required = ["scaler.pkl", "isolation_forest.pkl", "random_forest.pkl"]
        for f in required:
            files_present[f] = os.path.exists(f)
        # XGBoost file might be named xgboost.pkl or xgboost_model.pkl
        xgb_file = None
        if os.path.exists("xgboost.pkl"):
            xgb_file = "xgboost.pkl"
        elif os.path.exists("xgboost_model.pkl"):
            xgb_file = "xgboost_model.pkl"
        else:
            files_present["xgboost.pkl (or xgboost_model.pkl)"] = False
        
        if not all(files_present.values()) or xgb_file is None:
            st.warning("⚠️ Some ML model files are missing:")
            for f, exists in files_present.items():
                if not exists:
                    st.warning(f"   - {f} not found")
            if xgb_file is None:
                st.warning("   - xgboost.pkl or xgboost_model.pkl not found")
            st.info("💡 ML predictions will be disabled.")
            return None, None, None, None
        
        scaler = joblib.load("scaler.pkl")
        iso_model = joblib.load("isolation_forest.pkl")
        rf_model = joblib.load("random_forest.pkl")
        xgb_model = joblib.load(xgb_file)
        return scaler, iso_model, rf_model, xgb_model
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None, None, None

# =====================================================================
#  PAGE LAYOUT
# =====================================================================
st.title("💧 Smart Water Quality Monitoring System")
st.markdown("---")

# =====================================================================
#  SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### 🎯 System Control")
    auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=True)
    if st.button("⟳ Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("### 📡 System Status")
    st.markdown("✅ **NodeMCU:** Online")
    st.markdown("✅ **ThingSpeak:** Connected")
    st.markdown("---")
    st.caption(f"Channel ID: {CHANNEL_ID}")

# =====================================================================
#  LOAD DATA
# =====================================================================
df = load_data()
if df.empty:
    st.warning("📡 No data loaded. Please check ThingSpeak connection.")
    st.stop()

# =====================================================================
#  LATEST READING
# =====================================================================
latest = df.iloc[-1]
temp = latest["Temperature"]
ph = latest["pH"]
tds = latest["TDS"]
turb = latest["Turbidity"]
timestamp = latest["created_at"]

# =====================================================================
#  SAFETY ASSESSMENT
# =====================================================================
reasons = []
is_safe = True
if ph < PH_MIN:
    reasons.append(f"pH too low ({ph:.2f}) – should be ≥ {PH_MIN}")
    is_safe = False
elif ph > PH_MAX:
    reasons.append(f"pH too high ({ph:.2f}) – should be ≤ {PH_MAX}")
    is_safe = False
if tds > TDS_MAX:
    reasons.append(f"TDS too high ({tds:.0f} ppm) – should be ≤ {TDS_MAX}")
    is_safe = False
if turb > TURB_MAX:
    reasons.append(f"Turbidity too high ({turb:.2f} NTU) – should be ≤ {TURB_MAX}")
    is_safe = False
if temp < TEMP_MIN:
    reasons.append(f"Temperature too low ({temp:.1f}°C) – should be ≥ {TEMP_MIN}")
    is_safe = False
elif temp > TEMP_MAX:
    reasons.append(f"Temperature too high ({temp:.1f}°C) – should be ≤ {TEMP_MAX}")
    is_safe = False

# =====================================================================
#  METRICS ROW
# =====================================================================
st.markdown('<div class="section-title">📊 Current Water Quality</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌡️ Temperature</div>
        <div class="metric-value">{temp:.1f}</div>
        <div class="metric-unit">°C</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🧪 pH</div>
        <div class="metric-value">{ph:.2f}</div>
        <div class="metric-unit">pH</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💧 TDS</div>
        <div class="metric-value">{tds:.0f}</div>
        <div class="metric-unit">ppm</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌫️ Turbidity</div>
        <div class="metric-value">{turb:.2f}</div>
        <div class="metric-unit">NTU</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
#  SAFETY RECOMMENDATION
# =====================================================================
st.markdown("---")
st.markdown('<div class="section-title">🛡️ Water Safety Recommendation</div>', unsafe_allow_html=True)
if is_safe:
    st.markdown(f"""
    <div class="safe-banner">
        ✅ WATER IS SAFE – All parameters within recommended limits.
        <br><span style="font-size:14px;font-weight:400;">
            Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="unsafe-banner">
        ⚠️ WATER IS UNSAFE – Immediate action recommended!
        <br><span style="font-size:14px;font-weight:400;">
            Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Reason(s) for unsafe status:**")
    for r in reasons:
        st.markdown(f'<span class="reason-text">• {r}</span>', unsafe_allow_html=True)

# =====================================================================
#  GAUGE CHARTS
# =====================================================================
st.markdown("---")
st.markdown('<div class="section-title">📊 Live Gauges</div>', unsafe_allow_html=True)
def create_gauge(value, title, min_val, max_val, safe_max=None, safe_min=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#a0aec0'}},
        number={'font': {'size': 24, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickfont': {'color': '#a0aec0'}},
            'bar': {'color': "#00b894" if (safe_min <= value <= safe_max) else "#e17055"} if safe_max else {},
            'bgcolor': 'rgba(255,255,255,0.05)',
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, safe_min], 'color': 'rgba(225,112,85,0.2)'} if safe_min else {},
                {'range': [safe_min, safe_max], 'color': 'rgba(0,206,201,0.2)'} if safe_min and safe_max else {},
                {'range': [safe_max, max_val], 'color': 'rgba(225,112,85,0.2)'} if safe_max else {}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': safe_max if safe_max else max_val
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#a0aec0'})
    return fig

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.plotly_chart(create_gauge(temp, "Temperature (°C)", 0, 50, TEMP_MAX, TEMP_MIN), use_container_width=True)
with g2:
    st.plotly_chart(create_gauge(ph, "pH", 0, 14, PH_MAX, PH_MIN), use_container_width=True)
with g3:
    st.plotly_chart(create_gauge(tds, "TDS (ppm)", 0, 1000, TDS_MAX, 0), use_container_width=True)
with g4:
    st.plotly_chart(create_gauge(turb, "Turbidity (NTU)", 0, 10, TURB_MAX, 0), use_container_width=True)

# =====================================================================
#  HISTORICAL TRENDS
# =====================================================================
st.markdown("---")
st.markdown('<div class="section-title">📈 Historical Trends</div>', unsafe_allow_html=True)
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=df["created_at"], y=df["Temperature"], name="Temperature (°C)", line=dict(color="#00b894", width=2)))
fig_trend.add_trace(go.Scatter(x=df["created_at"], y=df["pH"], name="pH", line=dict(color="#0984e3", width=2), yaxis="y2"))
fig_trend.add_trace(go.Scatter(x=df["created_at"], y=df["TDS"], name="TDS (ppm)", line=dict(color="#fdcb6e", width=2), yaxis="y3"))
fig_trend.add_trace(go.Scatter(x=df["created_at"], y=df["Turbidity"], name="Turbidity (NTU)", line=dict(color="#e17055", width=2), yaxis="y4"))
fig_trend.update_layout(
    height=400, margin=dict(l=20, r=20, t=30, b=20),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font={'color': '#a0aec0'},
    xaxis={'title': 'Time', 'gridcolor': 'rgba(255,255,255,0.05)'},
    yaxis={'title': 'Temperature (°C) / pH', 'gridcolor': 'rgba(255,255,255,0.05)'},
    yaxis2={'title': 'TDS (ppm)', 'gridcolor': 'rgba(255,255,255,0.05)', 'overlaying': 'y', 'side': 'right'},
    yaxis3={'title': 'Turbidity (NTU)', 'gridcolor': 'rgba(255,255,255,0.05)', 'overlaying': 'y', 'side': 'right'},
    legend={'font': {'color': '#a0aec0'}, 'bgcolor': 'rgba(0,0,0,0.3)'}, hovermode='x unified'
)
st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================================
#  MACHINE LEARNING PREDICTIONS
# =====================================================================
scaler, iso_model, rf_model, xgb_model = load_ml_models()
if all([scaler, iso_model, rf_model, xgb_model]):
    st.markdown("---")
    st.markdown('<div class="section-title">🤖 Machine Learning Predictions</div>', unsafe_allow_html=True)
    features = df[["pH", "TDS", "Turbidity", "Temperature"]]
    X_scaled = scaler.transform(features)
    df["Anomaly"] = iso_model.predict(X_scaled)
    df["Water_Class"] = rf_model.predict(X_scaled)
    df["XGBoost"] = xgb_model.predict(X_scaled)
    latest_features = scaler.transform(features.tail(1))
    anomaly = iso_model.predict(latest_features)[0]
    quality = rf_model.predict(latest_features)[0]
    xgb = xgb_model.predict(latest_features)[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        if anomaly == 1:
            st.success("✅ **Normal**")
        else:
            st.error("⚠️ **Anomaly Detected**")
    with col2:
        status = "Unsafe" if quality == 1 else "Safe"
        color = "#e17055" if quality == 1 else "#00b894"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);padding:15px;border-radius:10px;text-align:center;">
            <div style="color:#a0aec0;font-size:14px;">Random Forest</div>
            <div style="color:{color};font-size:24px;font-weight:700;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        status = "Unsafe" if xgb == 1 else "Safe"
        color = "#e17055" if xgb == 1 else "#00b894"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);padding:15px;border-radius:10px;text-align:center;">
            <div style="color:#a0aec0;font-size:14px;">XGBoost</div>
            <div style="color:{color};font-size:24px;font-weight:700;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    with st.expander("🔍 Detailed Predictions"):
        st.dataframe(df[["created_at", "pH", "TDS", "Turbidity", "Temperature", "Anomaly", "Water_Class", "XGBoost"]].tail(30), use_container_width=True)
else:
    st.info("💡 ML models not found or incomplete. Place .pkl files in the same folder.")

# =====================================================================
#  DATA TABLE
# =====================================================================
with st.expander("📋 Raw Data"):
    st.dataframe(df[["created_at", "Temperature", "pH", "TDS", "Turbidity"]].tail(50), use_container_width=True)

# =====================================================================
#  AUTO-REFRESH
# =====================================================================
if auto_refresh:
    time.sleep(10)
    st.rerun()
