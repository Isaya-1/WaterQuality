import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib
import time
from datetime import datetime

# ---------- Load Models and Scaler ----------
scaler = joblib.load('scaler.pkl')
iso_model = joblib.load('isolation_forest.pkl')
rf_model = joblib.load('random_forest.pkl')
xgb_model = joblib.load('xgboost.pkl')

# ---------- Initialize Firebase ----------
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://waterqualitydata-692a7-default-rtdb.firebaseio.com/'
})

# ---------- Helper: Fetch Latest Reading ----------
def fetch_latest_reading():
    ref = db.reference('/water_quality_readings')
    # Get the most recent entry (by key order)
    all_readings = ref.order_by_key().limit_to_last(1).get()
    for key, value in all_readings.items():
        return value
    return None

# ---------- Helper: Push Alert to Firebase ----------
def push_alert(alert_type, message, sensor_data):
    alert_ref = db.reference('/alerts')
    alert_data = {
        'timestamp': datetime.now().isoformat(),
        'type': alert_type,
        'message': message,
        'sensor_data': sensor_data
    }
    alert_ref.push(alert_data)
    print(f"🚨 ALERT pushed: {message}")

# ---------- Main Loop ----------
print("🟢 Water Quality Monitor Started. Waiting for data...")
last_timestamp = None

while True:
    try:
        data = fetch_latest_reading()
        if not data:
            time.sleep(3)
            continue

        # Optional: avoid re-processing the same reading
        current_ts = data.get('timestamp', '')
        if current_ts == last_timestamp:
            time.sleep(3)
            continue
        last_timestamp = current_ts

        # Prepare DataFrame for prediction
        new_df = pd.DataFrame([data])
        feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
        X_new = new_df[feature_cols].values
        X_scaled = scaler.transform(X_new)

        # Run models
        anomaly = iso_model.predict(X_scaled)[0]          # -1 = anomaly
        rf_pred = rf_model.predict(X_scaled)[0]           # 1 = unsafe
        rf_proba = rf_model.predict_proba(X_scaled)[0][1] # probability of unsafe
        xgb_pred = xgb_model.predict(X_scaled)[0]         # 1 = unsafe

        # Print to console
        print(f"\n📊 New reading @ {data.get('timestamp', 'N/A')}")
        print(f"   pH={data['pH']:.2f}, TDS={data['TDS_ppm']:.1f}, Turb={data['Turbidity_NTU']:.2f}, Temp={data['Temperature_C']:.1f}")
        print(f"   → Anomaly: {'⚠️ YES' if anomaly == -1 else 'NO'}")
        print(f"   → Random Forest: {'🔥 UNSAFE' if rf_pred == 1 else '✅ SAFE'} (prob: {rf_proba:.2%})")
        print(f"   → XGBoost: {'🔥 UNSAFE' if xgb_pred == 1 else '✅ SAFE'}")

        # Decision logic: generate alert if unsafe or anomaly
        if anomaly == -1 or rf_pred == 1 or xgb_pred == 1:
            reasons = []
            if anomaly == -1:
                reasons.append("Isolation Forest detected anomaly pattern")
            if rf_pred == 1:
                reasons.append(f"Random Forest: unsafe (prob {rf_proba:.2%})")
            if xgb_pred == 1:
                reasons.append("XGBoost: unsafe")
            message = " | ".join(reasons)
            push_alert("CRITICAL", message, data)

        time.sleep(5)   # check every 5 seconds

    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(5)