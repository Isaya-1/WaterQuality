import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb

# Load water quality data - skip first row (title), use second row as header
water_file = r'C:\Users\CHOGORO\Desktop\WaterQualityModel\processed_water_quality_dataset1.csv'
df = pd.read_csv(water_file, skiprows=[0])  # skip the title row

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 3 rows:")
print(df.head(3))

# Encode target: Safe -> 0, Unsafe -> 1
le = LabelEncoder()
df['risk'] = le.fit_transform(df['Label'])  # Safe=0, Unsafe=1

# Features
feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
X = df[feature_cols].values
y = df['risk'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Class distribution:")
print(df['risk'].value_counts())

# Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = iso.fit_predict(X_scaled)
df['anomaly_flag'] = (df['anomaly'] == -1).astype(int)
print(f"Anomalies found: {df['anomaly_flag'].sum()} out of {len(df)}")

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost ---")
print(classification_report(y_test, y_pred_xgb))
print(f"F1-score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_xgb):.4f}")