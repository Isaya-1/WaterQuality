import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Path to the folder containing all CSV files
data_folder = r'C:\Users\CHOGORO\Desktop\WaterQualityModel\Datasets'

def load_district_data(district_num):
    # 1. Load all climate files for this district
    climate_files = [
        f'climate_data_embedde_District_{district_num}_Water_sources_data_MP_transformed.csv',
        f'climate_data_embedded_District_{district_num}_Toilet_quality_data_MP_transformed.csv',
        f'climate_data_embedded_District_{district_num}_Waste_management_facilities_data_MP_transformed.csv'
    ]
    
    df_climate_list = []
    for f in climate_files:
        path = os.path.join(data_folder, f)
        if os.path.exists(path):
            df_temp = pd.read_csv(path)
            df_climate_list.append(df_temp)
    
    if not df_climate_list:
        return None
    df_climate = pd.concat(df_climate_list, ignore_index=True)
    
    # 2. Load disease cases
    disease_file = f'District_{district_num}_Disease_cases_with_location_uuid_transformed.csv'
    disease_path = os.path.join(data_folder, disease_file)
    df_disease = pd.read_csv(disease_path)
    
    # 3. Aggregate disease cases by location + month
    df_disease_agg = df_disease.groupby(['Transformed_Latitude', 'Transformed_Longitude', 'Month'])['JUMLA_1m'].sum().reset_index()
    df_disease_agg.rename(columns={'JUMLA_1m': 'total_cases'}, inplace=True)
    
    print(f"District {district_num} disease file shape: {df_disease.shape}")
    print("Sample JUMLA_1m values:")
    print(df_disease['JUMLA_1m'].head(10))
    print("Sum of JUMLA_1m:", df_disease['JUMLA_1m'].sum())
    print("Unique months in disease file:", sorted(df_disease['Month'].unique()))
    print("Aggregated disease cases (first 10 rows):")
    print(df_disease_agg.head(10))
    print("Sum of total_cases after aggregation:", df_disease_agg['total_cases'].sum())

    # 4. Round coordinates to 3 decimals to increase chance of matching
    df_climate['Lat_round'] = df_climate['Transformed_Latitude'].round(3)
    df_climate['Lon_round'] = df_climate['Transformed_Longitude'].round(3)
    df_disease_agg['Lat_round'] = df_disease_agg['Transformed_Latitude'].round(3)
    df_disease_agg['Lon_round'] = df_disease_agg['Transformed_Longitude'].round(3)
    
    # 5. Merge on rounded coordinates + Month
    merged = pd.merge(df_climate, df_disease_agg,
                      on=['Lat_round', 'Lon_round', 'Month'],
                      how='left')
    
    # If no matches (all total_cases NaN), fallback to merge on Month only
    if merged['total_cases'].isnull().all():
        print(f"District {district_num}: No matches on coordinates. Using Month-only merge.")
        merged = pd.merge(df_climate, df_disease_agg, on=['Month'], how='left')
    
    # Clean up temporary columns
    merged.drop(['Lat_round', 'Lon_round'], axis=1, errors='ignore', inplace=True)
    
    # Fill missing with 0
    merged['total_cases'] = merged['total_cases'].fillna(0)
    merged['high_risk'] = (merged['total_cases'] > 0).astype(int)
    
    # Drop any remaining columns that are purely from disease side (except total_cases, high_risk)
    # But keep everything for now.
    
    return merged

# Load all districts
all_data = []
for district in range(1, 6):
    df_dist = load_district_data(district)
    if df_dist is not None:
        all_data.append(df_dist)
        print(f"District {district} loaded: {df_dist.shape}")

if not all_data:
    print("No data loaded. Check file paths.")
    exit()

master_df = pd.concat(all_data, ignore_index=True)
print(f"Master dataset shape: {master_df.shape}")

# Check class distribution
print("Class distribution of high_risk:")
print(master_df['high_risk'].value_counts())
print("Rows with total_cases > 0:", (master_df['total_cases'] > 0).sum())

# If still only zeros, we cannot train classifiers meaningfully
if master_df['high_risk'].nunique() < 2:
    print("WARNING: No positive disease cases found. Cannot train Random Forest or XGBoost.")
    print("You may need to adjust merging logic or use a different dataset.")
    # Still train Isolation Forest for demonstration
    # But skip supervised models
    train_anomaly_only = True
else:
    train_anomaly_only = False

# Climate columns
climate_cols = ['10u','10v','2d','2t','evabs','evaow','evatc','evavt','albedo',
                'lshf','lai_hv','lai_lv','pev','ro','src','skt','es','stl1',
                'stl2','stl3','stl4','ssro','slhf','ssr','str','sp','sro',
                'sshf','ssrd','strd','e','tp','swvl1','swvl2','swvl3','swvl4']

existing_climate_cols = [col for col in climate_cols if col in master_df.columns]
master_df_clean = master_df.dropna(subset=existing_climate_cols)
print(f"Rows after dropping missing climate data: {len(master_df_clean)}")

# Isolation Forest (always train)
scaler = StandardScaler()
X_climate = scaler.fit_transform(master_df_clean[existing_climate_cols])
iso_model = IsolationForest(contamination=0.1, random_state=42)
master_df_clean['anomaly_iso'] = iso_model.fit_predict(X_climate)
master_df_clean['anomaly_flag'] = (master_df_clean['anomaly_iso'] == -1).astype(int)
master_df_clean['anomaly_score'] = iso_model.decision_function(X_climate)
print(f"Isolation Forest done. Anomalies found: {master_df_clean['anomaly_flag'].sum()}")

# If no positive cases, stop here
if train_anomaly_only:
    print("Skipping Random Forest and XGBoost due to lack of positive disease cases.")
    exit()

# Prepare for supervised learning
feature_cols = existing_climate_cols + ['anomaly_score']
X = master_df_clean[feature_cols]
y = master_df_clean['high_risk']

# Chronological split if Year column exists
if 'Year' in master_df_clean.columns:
    master_df_clean = master_df_clean.sort_values(['Year', 'Month'])
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler_ml = StandardScaler()
X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf))
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# XGBoost
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=scale_pos)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
print("\n--- XGBoost Results ---")
print(classification_report(y_test, y_pred_xgb))
print(f"F1-score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_xgb):.4f}")