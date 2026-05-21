import pandas as pd
import os

data_folder = r'C:\Users\CHOGORO\Desktop\WaterQualityModel'

# List files (optional, but useful)
print("Files in folder:")
for f in os.listdir(data_folder):
    print(f)

# 1. Load Busia disease data (it's a CSV file)
busia_file = os.path.join(data_folder, 'busia-county-waterborn-diseases-jan-2019-to-april-2020..xlsx-sheet3.csv')
busia_data = pd.read_csv(busia_file)
print("Busia data shape:", busia_data.shape)
print("Busia columns:", busia_data.columns.tolist())

# 2. Load water quality CSV (already a CSV)
water_file = os.path.join(data_folder, 'processed_water_quality_dataset1.csv')
# Try different encodings if needed
encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
water_quality = None
for enc in encodings:
    try:
        water_quality = pd.read_csv(water_file, encoding=enc)
        print(f"Loaded water quality with encoding: {enc}")
        break
    except UnicodeDecodeError:
        continue
if water_quality is None:
    raise ValueError("Could not read water quality CSV with any common encoding.")

print("Water quality shape:", water_quality.shape)
print("Water quality columns:", water_quality.columns.tolist())