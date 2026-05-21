import pandas as pd
import joblib

# Load models
scaler = joblib.load('scaler.pkl')
rf = joblib.load('random_forest.pkl')
iso = joblib.load('isolation_forest.pkl')

# Load first 100 rows
df = pd.read_csv(r'C:\Users\CHOGORO\Desktop\WaterQualityModel\processed_water_quality_dataset1.csv', skiprows=[0], nrows=100)
feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
X_raw = df[feature_cols].values
X_scaled = scaler.transform(X_raw)

# Predict
rf_preds = rf.predict(X_scaled)
iso_preds = iso.predict(X_scaled)

# Compare with true labels
true_unsafe = (df['Label'] == 'Unsafe').astype(int).values

print("First 20 predictions vs true:")
for i in range(20):
    print(f"Row {i}: RF={rf_preds[i]}, True={true_unsafe[i]}, Anomaly={'Yes' if iso_preds[i]==-1 else 'No'}")

print(f"\nTotal unsafe in first 100: {true_unsafe.sum()}")
print(f"Total RF predictions as unsafe: {rf_preds.sum()}")