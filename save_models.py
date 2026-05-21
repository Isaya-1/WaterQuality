import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv(r'C:\Users\CHOGORO\Desktop\WaterQualityModel\processed_water_quality_dataset1.csv', skiprows=[0])
df['risk'] = (df['Label'] == 'Unsafe').astype(int)
feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
X = df[feature_cols].values
y = df['risk'].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_scaled, y)

# Save
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(iso, 'isolation_forest.pkl')
joblib.dump(rf, 'random_forest.pkl')
joblib.dump(xgb_model, 'xgboost.pkl')

print("All models saved successfully!")