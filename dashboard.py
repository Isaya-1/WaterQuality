import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt


# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="Water Quality Monitoring",
    page_icon="💧",
    layout="wide"
)


st.title("💧 IoT Water Quality Monitoring Dashboard")

st.write(
"""
ESP8266 + CD74HC4067 Sensor System
with Machine Learning Anomaly Detection
"""
)



# ============================
# THINGSPEAK CONFIG
# ============================


CHANNEL_ID = "YOUR_CHANNEL_ID"

READ_API_KEY = "YOUR_READ_API_KEY"


URL = (
f"https://api.thingspeak.com/channels/"
f"{CHANNEL_ID}/feeds.json?"
f"api_key={READ_API_KEY}&results=200"
)



# ============================
# LOAD DATA
# ============================

@st.cache_data(ttl=60)
def load_data():

    response = requests.get(URL)

    data = response.json()

    feeds = data["feeds"]


    df = pd.DataFrame(feeds)


    df["created_at"] = pd.to_datetime(
        df["created_at"]
    )


    df.rename(
        columns={
            "field1":"Temperature",
            "field2":"pH",
            "field3":"TDS",
            "field4":"Turbidity"
        },
        inplace=True
    )


    columns=[
        "Temperature",
        "pH",
        "TDS",
        "Turbidity"
    ]


    for col in columns:
        df[col]=pd.to_numeric(
            df[col],
            errors="coerce"
        )


    df.dropna(
        inplace=True
    )


    return df



df = load_data()



if df.empty:

    st.warning(
        "No sensor data available"
    )

    st.stop()



# ============================
# DISPLAY DATA
# ============================


st.subheader("Latest Sensor Data")


st.dataframe(
    df.tail(10),
    use_container_width=True
)




# ============================
# SENSOR CHARTS
# ============================


st.subheader("Sensor Trends")


col1,col2 = st.columns(2)


with col1:

    st.write("Temperature")

    st.line_chart(
        df.set_index("created_at")
        ["Temperature"]
    )


    st.write("pH")

    st.line_chart(
        df.set_index("created_at")
        ["pH"]
    )



with col2:

    st.write("TDS")

    st.line_chart(
        df.set_index("created_at")
        ["TDS"]
    )


    st.write("Turbidity")

    st.line_chart(
        df.set_index("created_at")
        ["Turbidity"]
    )





# ============================
# MACHINE LEARNING
# ISOLATION FOREST
# ============================


st.subheader(
    "🤖 Machine Learning Anomaly Detection"
)



features = df[
[
"Temperature",
"pH",
"TDS",
"Turbidity"
]
]



model = IsolationForest(

    contamination=0.1,
    random_state=42

)



model.fit(features)



df["Anomaly"] = model.predict(
    features
)



df["Status"] = df["Anomaly"].apply(

lambda x:

"Normal" if x==1 else "⚠️ Anomaly"

)



# ============================
# RESULTS
# ============================


normal = len(
df[df["Anomaly"]==1]
)


abnormal = len(
df[df["Anomaly"]==-1]
)



col1,col2 = st.columns(2)


with col1:

    st.metric(
        "Normal Samples",
        normal
    )


with col2:

    st.metric(
        "Detected Anomalies",
        abnormal
    )




st.subheader(
    "Detection Results"
)


st.dataframe(

df[
[
"created_at",
"Temperature",
"pH",
"TDS",
"Turbidity",
"Status"
]
].tail(20),

use_container_width=True

)




# ============================
# CURRENT WATER STATUS
# ============================


latest=df.iloc[-1]


st.subheader(
    "Current Water Status"
)



if latest["Status"]=="Normal":

    st.success(
        "Water quality parameters are within normal pattern"
    )

else:

    st.error(
        "⚠️ Possible abnormal water condition detected"
    )


