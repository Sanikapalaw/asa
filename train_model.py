# ==========================================
# Urban Logistics - Deep Learning Framework
# ==========================================

import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. Load Dataset
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Zomato Dataset.csv")

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("\nDataset Shape:", df.shape)

# ==========================================
# 2. Auto Detect Target Column
# ==========================================

possible_targets = [col for col in df.columns if "time" in col.lower()]

if len(possible_targets) == 0:
    raise Exception("No time-related column found.")

TARGET_COLUMN = possible_targets[-1]

print(f"Using Target Column: {TARGET_COLUMN}")

# Clean target column
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)
df[TARGET_COLUMN] = df[TARGET_COLUMN].str.extract(r"(\d+)")
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)
df = df.dropna(subset=[TARGET_COLUMN])

# ==========================================
# 3. Drop Unnecessary Columns
# ==========================================

drop_cols = ["ID", "Delivery_person_ID"]
for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ==========================================
# 4. Handle Missing Values
# ==========================================

df = df.ffill()

# ==========================================
# 5. Encode Categorical Features
# ==========================================

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ==========================================
# 6. Split Features & Target
# ==========================================

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 7. Random Forest Baseline Model
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("\n===== Random Forest Results =====")
print("MAE:", rf_mae)
print("R2:", rf_r2)

# Save Feature Importance
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

# ==========================================
# 8. Scale Data for Deep Learning
# ==========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 9. Deep Learning Model
# ==========================================

dl_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1)
])

dl_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = dl_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

dl_preds = dl_model.predict(X_test_scaled)

dl_mae = mean_absolute_error(y_test, dl_preds)
dl_r2 = r2_score(y_test, dl_preds)

print("\n===== Deep Learning Results =====")
print("MAE:", dl_mae)
print("R2:", dl_r2)

# ==========================================
# 10. Save Models & Artifacts
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")
dl_model.save("models/dl_model.keras")

# Save metrics
metrics = {
    "rf_mae": float(rf_mae),
    "rf_r2": float(rf_r2),
    "dl_mae": float(dl_mae),
    "dl_r2": float(dl_r2)
}

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f)

# Save feature importance
feature_importance.to_csv("models/feature_importance.csv", index=False)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv("models/training_history.csv", index=False)

print("\nModels and evaluation artifacts saved successfully.")