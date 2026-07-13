import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time

# =====================================================================
#  CONFIGURATION – REPLACE WITH YOUR THINGSPEAK CHANNEL DETAILS
# =====================================================================
CHANNEL_ID = "3389783"                      # Your ThingSpeak Channel ID
READ_API_KEY = "04JJ8T8BA0TIIBQ0"           # Your Read API Key

# =====================================================================
#  SAFETY THRESHOLDS (WHO / Tanzania standards)
# =====================================================================
PH_MIN, PH_MAX = 6.5, 8.5
TDS_MAX = 500
TURB_MAX = 5.0
TEMP_MIN, TEMP_MAX = 15, 35

# =====================================================================
#  FETCH DATA FROM THINGSPEAK
# =====================================================================
@st.cache_data(ttl=10)
def load_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=100"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            st.error(f"ThingSpeak error: HTTP {r.status_code}")
            return pd.DataFrame()
        data = r.json()
        feeds = data.get("feeds", [])
        if not feeds:
            st.warning("No data found.")
            return pd.DataFrame()
        df = pd.DataFrame(feeds)

        # Correct column mapping from ESP8266/ESP32 code:
        # field1 = pH, field2 = TDS, field3 = Turbidity, field4 = Temperature
        df.rename(columns={
            "field1": "pH",
            "field2": "TDS",
            "field3": "Turbidity",
            "field4": "Temperature"
        }, inplace=True)

        # Convert to datetime and numeric
        df["created_at"] = pd.to_datetime(df["created_at"])
        for col in ["pH", "TDS", "Turbidity", "Temperature"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# =====================================================================
#  PAGE CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Water Quality Monitor", page_icon="💧", layout="wide")
st.title("💧 Smart Water Quality Monitoring System")
st.markdown("---")

# =====================================================================
#  LOAD DATA
# =====================================================================
df = load_data()
if df.empty:
    st.warning("No data loaded. Please check your ThingSpeak channel and internet connection.")
    st.stop()

# =====================================================================
#  LATEST READING
# =====================================================================
latest = df.iloc[-1]
ph = latest["pH"]
tds = latest["TDS"]
turb = latest["Turbidity"]
temp = latest["Temperature"]

# =====================================================================
#  SAFETY ASSESSMENT (threshold-based)
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
#  METRICS ROW (ORDER: Temperature, Turbidity, TDS, pH)
# =====================================================================
st.subheader("📊 Current Water Quality")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🌡️ Temperature", f"{temp:.1f} °C")
with c2:
    st.metric("🌫️ Turbidity", f"{turb:.2f} NTU")
with c3:
    st.metric("💧 TDS", f"{tds:.0f} ppm")
with c4:
    st.metric("🧪 pH", f"{ph:.2f}")

# =====================================================================
#  SAFETY RECOMMENDATION
# =====================================================================
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

# =====================================================================
#  GAUGE CHARTS (ORDER: Temperature, Turbidity, TDS, pH)
# =====================================================================
st.markdown("---")
st.subheader("📊 Live Gauges")

def create_gauge(value, title, min_val, max_val, safe_max=None, safe_min=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "green" if (safe_min <= value <= safe_max) else "red"} if safe_max else {},
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
    st.plotly_chart(create_gauge(temp, "Temperature (°C)", 0, 50, TEMP_MAX, TEMP_MIN), use_container_width=True)
with g2:
    st.plotly_chart(create_gauge(turb, "Turbidity (NTU)", 0, 10, TURB_MAX, 0), use_container_width=True)
with g3:
    st.plotly_chart(create_gauge(tds, "TDS (ppm)", 0, 1000, TDS_MAX, 0), use_container_width=True)
with g4:
    st.plotly_chart(create_gauge(ph, "pH", 0, 14, PH_MAX, PH_MIN), use_container_width=True)

# =====================================================================
#  HISTORICAL TRENDS (ORDER: Temperature, Turbidity, TDS, pH)
# =====================================================================
st.markdown("---")
st.subheader("📈 Historical Trends")
st.line_chart(df[["Temperature", "Turbidity", "TDS", "pH"]])

# =====================================================================
#  OPTIONAL: MACHINE LEARNING PREDICTIONS (if models exist)
# =====================================================================
try:
    import joblib
    iso_model = joblib.load("isolation_forest.pkl")
    rf_model = joblib.load("random_forest.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    models_loaded = True
except (FileNotFoundError, ImportError):
    models_loaded = False

if models_loaded:
    st.markdown("---")
    st.subheader("🤖 Machine Learning Predictions")

    # Features in correct order for model: Temperature, pH, TDS, Turbidity
    features = df[["Temperature", "pH", "TDS", "Turbidity"]]
    df["Anomaly"] = iso_model.predict(features)
    df["Water_Class"] = rf_model.predict(features)
    df["XGBoost"] = xgb_model.predict(features)

    latest_features = features.tail(1)
    anomaly = iso_model.predict(latest_features)[0]
    quality = rf_model.predict(latest_features)[0]
    xgb = xgb_model.predict(latest_features)[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        if anomaly == 1:
            st.success("✅ Normal")
        else:
            st.error("⚠️ Anomaly Detected")
    with col2:
        st.metric("Random Forest", "Safe" if quality == 1 else "Unsafe")
    with col3:
        st.metric("XGBoost", "Safe" if xgb == 1 else "Unsafe")

    with st.expander("🔍 Detailed Predictions"):
        st.dataframe(df[["created_at", "Temperature", "Turbidity", "TDS", "pH", "Anomaly", "Water_Class", "XGBoost"]].tail(20),
                     use_container_width=True)
else:
    st.info("💡 ML models not found. To enable predictions, place your trained .pkl files in the same folder.")

# =====================================================================
#  AUTO-REFRESH (optional)
# =====================================================================
if st.sidebar.checkbox("🔄 Auto‑refresh (5 seconds)", value=True):
    time.sleep(5)
    st.rerun()
