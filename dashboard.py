import streamlit as st
import pandas as pd
import requests
import time

# ========== MIPANGILIO YA THINGSPEAK ==========
CHANNEL_ID = "3389783"  # Tafuta kwenye Channel Settings ya ThingSpeak
READ_API_KEY = "04JJ8T8BA0TIIBQ0"  # Tafuta kwenye kichupo cha API Keys

# Ikiwa chaneli yako ni ya umma (public), unaweza kutumia URL hii isiyo na Read API Key:
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?results=30"

# ========== USANIFU WA DASHBOARD ==========
st.set_page_config(page_title="Mfumo wa Ubora wa Maji", layout="wide")
st.title("🌊 Mfumo wa Kufuatilia Ubora wa Maji")
st.markdown("---")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Udhibiti")
    auto_refresh = st.checkbox("Ongeza upya kiotomatiki (kila sekunde 5)", value=True)
    st.caption("Data inasomwa moja kwa moja kutoka ThingSpeak")

# ========== KAZI YA KUSOMA DATA ==========
@st.cache_data(ttl=5)
def fetch_data():
    try:
        response = requests.get(THINGSPEAK_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            feeds = data['feeds']
            if feeds:
                df = pd.DataFrame(feeds)
                # Badilisha maadili ya namba kuwa namba (kwa usalama)
                for col in ['field1', 'field2', 'field3', 'field4']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                # Badilisha wakati kuwa muundo unaoeleweka
                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                return df
            else:
                return pd.DataFrame()
        else:
            st.error(f"ThingSpeak API imeshindwa: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Hitilafu ya mtandao: {e}")
        return pd.DataFrame()

# ========== ONYESHA DATA ==========
df = fetch_data()

if df.empty:
    st.warning("⚠️ Hakuna data bado. Subiri NodeMCU itume data kwa ThingSpeak.")
    st.info("Tafadhali angalia Serial Monitor ya Arduino IDE kuhakikisha kuwa data inatumwa.")
else:
    # Pata usomaji wa mwisho kabisa
    latest = df.iloc[-1]
    
    # Onyesha maadili ya sasa kwa kutumia vifuniko (metrics)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧪 pH", f"{latest['field1']:.2f}" if pd.notna(latest['field1']) else "N/A")
    col2.metric("💧 TDS", f"{latest['field2']:.0f} ppm" if pd.notna(latest['field2']) else "N/A")
    col3.metric("🌫️ Turbidity", f"{latest['field3']:.2f} NTU" if pd.notna(latest['field3']) else "N/A")
    col4.metric("🌡️ Joto", f"{latest['field4']:.1f} °C" if pd.notna(latest['field4']) else "N/A")
    
    # ========== HALI YA USALAMA WA MAJI (HIARI) ==========
    st.markdown("---")
    st.subheader("🛡️ Hali ya Usalama wa Maji")
    
    # Badilisha maadili haya kulingana na viwango vya WHO au mahitaji yako
    PH_MIN, PH_MAX = 6.5, 8.5
    TDS_MAX = 500
    TURB_MAX = 5.0
    
    is_safe = True
    reasons = []
    
    if not (PH_MIN <= latest['field1'] <= PH_MAX):
        is_safe = False
        reasons.append(f"pH ipo nje ya kiwango ({latest['field1']:.2f})")
    if latest['field2'] > TDS_MAX:
        is_safe = False
        reasons.append(f"TDS ni juu sana ({latest['field2']:.0f} ppm)")
    if latest['field3'] > TURB_MAX:
        is_safe = False
        reasons.append(f"Turbidity ni juu sana ({latest['field3']:.2f} NTU)")
    
    if is_safe:
        st.success("✅ MAJI NI SALAMA KUNYWA!")
    else:
        st.error("⚠️ MAJI SI SALAMA! Hatua za tahadhari zinahitajika.")
        for reason in reasons:
            st.warning(f"- {reason}")
    
    # ========== GRAFU ZA HISTORIA ==========
    st.markdown("---")
    st.subheader("📈 Historia ya Mabadiliko ya pH na Joto")
    
    # Ondoa thamani zisizo sahihi (NaN)
    plot_df = df.dropna(subset=['created_at', 'field1', 'field4'])
    if not plot_df.empty:
        plot_df = plot_df.set_index('created_at')
        st.line_chart(plot_df[['field1', 'field4']])
    else:
        st.info("Hakuna data ya kutosha kwa ajili ya grafu.")

# ========== MWISHO WA MFUMO WA KUJIUNGANISHA (AUTO-REFRESH) ==========
if auto_refresh:
    time.sleep(5)
    st.rerun()