import pandas as pd

water_file = r'C:\Users\CHOGORO\Desktop\WaterQualityModel\processed_water_quality_dataset1.csv'

# Read without assuming header, to see the raw first row
water_raw = pd.read_csv(water_file, header=None, nrows=5)
print("First 5 rows (raw):")
print(water_raw)