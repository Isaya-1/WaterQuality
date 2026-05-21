import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from time import sleep
from collections import deque

# -------------------------------
# 1. Load models and scaler
# -------------------------------
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('random_forest.pkl')
xgb_model = joblib.load('xgboost.pkl')
iso_model = joblib.load('isolation_forest.pkl')

# Load LSTM if you have it
try:
    lstm_model = load_model('lstm_model.h5')
    use_lstm = True
except:
    use_lstm = False
    print("LSTM model not found. Proceeding without LSTM.")

# -------------------------------
# 2. Load the water quality CSV (to replay)
# -------------------------------
data_file = r'C:\Users\CHOGORO\Desktop\WaterQualityModel\processed_water_quality_dataset1.csv'
df = pd.read_csv(data_file, skiprows=[0])

# We'll iterate over rows (excluding header)
# You can limit to first N rows for testing: df = df.head(1000)

# Column order expected by scaler (must match training)
feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']

# For LSTM buffer (need last 15 scaled samples)
if use_lstm:
    buffer = deque(maxlen=15)
    lstm_pred = None

# -------------------------------
# 3. Replay each row as a "live" reading
# -------------------------------
print("Starting real‑time water quality monitoring simulation...")
print("Press Ctrl+C to stop.\n")

for idx, row in df.iterrows():
    # Extract sensor values
    raw_values = [row[col] for col in feature_cols]
    
    # Scale the vector
    scaled = scaler.transform([raw_values])[0]
    
    # --- Isolation Forest anomaly detection ---
    iso_pred = iso_model.predict([scaled])[0]   # -1 = anomaly, 1 = normal
    is_anomaly = (iso_pred == -1)
    
    # --- Random Forest / XGBoost safety prediction ---
    # Use RF or XGB; we can use both and compare
    rf_pred = rf_model.predict([scaled])[0]     # 0 = Safe, 1 = Unsafe
    xgb_pred = xgb_model.predict([scaled])[0]
    # For simplicity, we'll say water is unsafe if either model says Unsafe
    unsafe = (rf_pred == 1) or (xgb_pred == 1)
    
    # --- LSTM (if available) ---
    if use_lstm:
        buffer.append(scaled)
        if len(buffer) == 15:
            input_seq = np.array(buffer).reshape(1, 15, len(feature_cols))
            lstm_out = lstm_model.predict(input_seq, verbose=0)[0][0]
            lstm_risk = lstm_out > 0.5   # threshold 0.5
        else:
            lstm_risk = None
    else:
        lstm_risk = None
    
    # --- Print results and alerts ---
    if is_anomaly:
        print(f"[ALERT] Anomaly detected at row {idx}: {raw_values}")
    if unsafe:
        print(f"[CRITICAL] UNSAFE water at row {idx}: {raw_values}")
        # You could send an SMS/email here
    if lstm_risk is True:
        print(f"[LSTM RISK] High risk predicted at row {idx}")
    
    # Optional: print a dot for normal operation
    if not is_anomaly and not unsafe:
        print(".", end="", flush=True)
    
    # Simulate sensor sampling interval (e.g., 0.1 seconds)
    sleep(0.1)

print("\nSimulation finished.")