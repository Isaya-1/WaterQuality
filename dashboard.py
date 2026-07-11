import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Water Quality Monitor", layout="wide")
st.title("🌊 Smart Water Quality Monitoring System")

# ========== THINGSPEAK CONFIGURATION ==========
CHANNEL_ID = "3389783"
READ_API_KEY = "04JJ8T8BA0TIIBQ0"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=30"

# ========== SAFETY THRESHOLDS ==========
PH_MIN, PH_MAX = 6.5, 8.5
TDS_MAX = 500
TURB_MAX = 5.0
TEMP_MIN, TEMP_MAX = 15, 35

# ========== FETCH DATA ==========
@st.cache_data(ttl=10)
def fetch_data():
    try:
        r = requests.get(THINGSPEAK_URL, timeout=5)
        if r.status_code == 200:
            data = r.json()
            feeds = data.get('feeds', [])
            if not feeds:
                return pd.DataFrame()
            df = pd.DataFrame(feeds)
            for col in ['field1', 'field2', 'field3', 'field4']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

df = fetch_data()

if df.empty:
    st.warning("📡 No data received yet. Waiting for NodeMCU...")
    st.info("Please check your ESP32 Serial Monitor and ThingSpeak channel.")
    st.stop()

# ========== LATEST READING ==========
latest = df.iloc[-1]
ph = latest.get('field1', 0.0)
tds = latest.get('field2', 0.0)
turb = latest.get('field3', 0.0)
temp = latest.get('field4', 0.0)

# ========== SAFETY ASSESSMENT ==========
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

# ========== METRICS ROW ==========
st.subheader("📊 Current Water Quality")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🧪 pH", f"{ph:.2f}")
col2.metric("💧 TDS", f"{tds:.0f} ppm")
col3.metric("🌫️ Turbidity", f"{turb:.2f} NTU")
col4.metric("🌡️ Temperature", f"{temp:.1f} °C")

# ========== SAFETY STATUS BANNER ==========
st.markdown("---")
st.subheader("🛡️ Water Safety Recommendation")

if is_safe:
    st.success("✅ **WATER IS SAFE** – All parameters within recommended limits.")
    st.info("👍 No action required. Continue monitoring.")
else:
    st.error("⚠️ **WATER IS UNSAFE** – Immediate action recommended!")
    st.warning("**Reason(s) for unsafe status:**")
    for r in reasons:
        st.write(f"• {r}")

# ========== GAUGE CHARTS ==========
st.markdown("---")
st.subheader("📊 Live Gauges")

def create_gauge(value, title, min_val, max_val, safe_max=None, safe_min=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "green" if (value <= safe_max and value >= safe_min) else "red"} if safe_max else {},
            'steps': [
                {'range': [min_val, safe_min], 'color': "lightcoral"} if safe_min else {},
                {'range': [safe_min, safe_max], 'color': "lightgreen"} if safe_min and safe_max else {},
                {'range': [safe_max, max_val], 'color': "lightcoral"} if safe_max else {}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': safe_max if safe_max else max_val
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.plotly_chart(create_gauge(ph, "pH", 0, 14, PH_MAX, PH_MIN), use_container_width=True)
with g2:
    st.plotly_chart(create_gauge(tds, "TDS (ppm)", 0, 1000, TDS_MAX, 0), use_container_width=True)
with g3:
    st.plotly_chart(create_gauge(turb, "Turbidity (NTU)", 0, 10, TURB_MAX, 0), use_container_width=True)
with g4:
    st.plotly_chart(create_gauge(temp, "Temperature (°C)", 0, 50, TEMP_MAX, TEMP_MIN), use_container_width=True)

# ========== HISTORICAL TRENDS ==========
st.markdown("---")
st.subheader("📈 Historical Trends")
st.line_chart(df[['field1', 'field2', 'field3', 'field4']])

# ========== AUTO-REFRESH ==========
if st.sidebar.checkbox("🔄 Auto-refresh (5 seconds)", value=True):
    time.sleep(5)
    st.rerun()
