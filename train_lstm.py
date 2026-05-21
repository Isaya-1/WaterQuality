import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and prepare data (same as before)
water_file = r'C:\Users\CHOGORO\Desktop\WaterQualityModel\processed_water_quality_dataset1.csv'
df = pd.read_csv(water_file, skiprows=[0])
df['risk'] = (df['Label'] == 'Unsafe').astype(int)

feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
y = df['risk'].values

# Create sequences (look back 10 steps)
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])  # predict next step's risk
    return np.array(X_seq), np.array(y_seq)

seq_len = 10
X_seq, y_seq = create_sequences(X_scaled, y, seq_len)
print(f"Sequences shape: {X_seq.shape}, Labels shape: {y_seq.shape}")

# Split (chronologically – first 80% train, last 20% test)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(seq_len, len(feature_cols)), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_split=0.2, callbacks=[early_stop], verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"LSTM Test Accuracy: {acc:.4f}")