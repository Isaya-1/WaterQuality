import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Water Quality", layout="wide")
st.title("🌊 Smart Water Quality Monitor")

# ThingSpeak settings (badilisha na yako)
CHANNEL_ID = "YOUR_CHANNEL_ID"
READ_API_KEY = "YOUR_READ_API_KEY"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=30"

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
    st.warning("No data yet. Waiting for NodeMCU...")
else:
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧪 pH", f"{latest['field1']:.2f}")
    c2.metric("💧 TDS", f"{latest['field2']:.0f} ppm")
    c3.metric("🌫️ Turbidity", f"{latest['field3']:.2f} NTU")
    c4.metric("🌡️ Temperature", f"{latest['field4']:.1f} °C")

    # Gauges
    def create_gauge(value, title, min_val, max_val, safe_max=None):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "green" if value <= safe_max else "red"} if safe_max else {},
                'steps': [
                    {'range': [min_val, safe_max], 'color': "lightgreen"} if safe_max else {},
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
    g1.plotly_chart(create_gauge(latest['field1'], "pH", 0, 14, 8.5), use_container_width=True)
    g2.plotly_chart(create_gauge(latest['field2'], "TDS (ppm)", 0, 1000, 500), use_container_width=True)
    g3.plotly_chart(create_gauge(latest['field3'], "Turbidity (NTU)", 0, 10, 5), use_container_width=True)
    g4.plotly_chart(create_gauge(latest['field4'], "Temperature (°C)", 0, 50, 35), use_container_width=True)

    st.subheader("📈 Historical Trends")
    st.line_chart(df[['field1', 'field2']])

    if st.sidebar.checkbox("Auto-refresh (5s)", True):
        time.sleep(5)
        st.rerun()
