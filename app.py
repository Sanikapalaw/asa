import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pydeck as pdk
import json
import requests
import os
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Urban Logistics AI", layout="wide")
st.title("🚚 Intelligent Urban Logistics Optimization System")
st.caption("Deep Learning-Based Strategic Last-Mile Delivery Decision Support")

# --------------------------------------------------
# LOAD MODELS (FROM ROOT FOLDER)
# --------------------------------------------------
@st.cache_resource
def load_dl_model():
    if not os.path.exists("dl_model.keras"):
        st.error("Model file not found!")
        st.stop()
    return load_model("dl_model.keras")

@st.cache_resource
def load_scaler():
    if not os.path.exists("scaler.pkl"):
        st.error("Scaler file not found!")
        st.stop()
    return joblib.load("scaler.pkl")

model = load_dl_model()
scaler = load_scaler()

# Feature names safe check
if hasattr(scaler, "feature_names_in_"):
    expected_features = scaler.feature_names_in_
else:
    expected_features = None

# Load metrics safely
try:
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
except:
    metrics = {}

# Load feature importance safely
try:
    feature_importance = pd.read_csv("feature_importance.csv")
except:
    feature_importance = pd.DataFrame()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("📍 Delivery Details")

store_lat = st.sidebar.number_input("Store Latitude", value=19.2089)
store_lon = st.sidebar.number_input("Store Longitude", value=72.8722)

customer_lat = st.sidebar.number_input("Customer Latitude", value=19.2108)
customer_lon = st.sidebar.number_input("Customer Longitude", value=72.8746)

store_rating = st.sidebar.slider("Store Rating", 1.0, 5.0, 4.0)
order_cost = st.sidebar.number_input("Order Cost (₹)", value=300)

traffic_level = st.sidebar.selectbox(
    "🚦 Traffic Level",
    ["Low", "Moderate", "High"]
)

# --------------------------------------------------
# SIMPLE DISTANCE CALCULATION (No API Dependency)
# --------------------------------------------------
distance = np.sqrt(
    (store_lat - customer_lat)**2 +
    (store_lon - customer_lon)**2
) * 111  # Rough km conversion

ors_duration = distance * 4  # Assume avg 15 km/hr city speed

# --------------------------------------------------
# PREPARE MODEL INPUT
# --------------------------------------------------
if expected_features is not None:
    input_dict = {}

    for feature in expected_features:

        if "latitude" in feature.lower() and "delivery" not in feature.lower():
            input_dict[feature] = store_lat

        elif "longitude" in feature.lower() and "delivery" not in feature.lower():
            input_dict[feature] = store_lon

        elif "delivery" in feature.lower() and "latitude" in feature.lower():
            input_dict[feature] = customer_lat

        elif "delivery" in feature.lower() and "longitude" in feature.lower():
            input_dict[feature] = customer_lon

        elif "distance" in feature.lower():
            input_dict[feature] = distance

        elif "rating" in feature.lower():
            input_dict[feature] = store_rating

        elif "cost" in feature.lower():
            input_dict[feature] = order_cost

        else:
            input_dict[feature] = 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_features]
    input_scaled = scaler.transform(input_df)

else:
    st.error("Scaler missing feature names.")
    st.stop()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
model_time = model.predict(input_scaled, verbose=0)[0][0]

prep_time = 10
logic_time = prep_time + ors_duration

predicted_time = (model_time * 0.6) + (logic_time * 0.4)

traffic_factor = {"Low":1.0, "Moderate":1.2, "High":1.5}[traffic_level]
optimized_time = predicted_time / traffic_factor

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
st.subheader("📊 Operational Prediction Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("📏 Road Distance (km)", f"{distance:.2f}")
col2.metric("⏱ Predicted Delivery Time (mins)", f"{predicted_time:.2f}")
col3.metric("🚀 Optimized Time (mins)", f"{optimized_time:.2f}")

sla_threshold = 40

if predicted_time > sla_threshold:
    st.error("⚠ High Delay Risk - Increase manpower")
elif predicted_time > 30:
    st.warning("⚠ Moderate Delay Risk - Monitor closely")
else:
    st.success("✔ Low Delay Risk - Operations Stable")

improvement = ((predicted_time - optimized_time) / predicted_time) * 100
st.write(f"📈 Estimated Efficiency Gain: {improvement:.2f}%")

# --------------------------------------------------
# MAP VISUALIZATION
# --------------------------------------------------
st.subheader("🗺 Delivery Route")

route_layer = pdk.Layer(
    "LineLayer",
    data=[{
        "start": [store_lon, store_lat],
        "end": [customer_lon, customer_lat]
    }],
    get_source_position="start",
    get_target_position="end",
    get_width=5,
    get_color=[255, 0, 0],
)

view_state = pdk.ViewState(
    latitude=(store_lat + customer_lat) / 2,
    longitude=(store_lon + customer_lon) / 2,
    zoom=12,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[route_layer],
        initial_view_state=view_state
    )
)

# --------------------------------------------------
# ADVANCED MODEL ANALYSIS
# --------------------------------------------------
with st.expander("📊 Advanced Model Evaluation"):

    if metrics:
        st.write("Deep Learning MAE:", round(metrics.get("dl_mae", 0), 2))
        st.write("Deep Learning R²:", round(metrics.get("dl_r2", 0), 2))

    if not feature_importance.empty:
        st.markdown("### Top Feature Importance")
        st.dataframe(feature_importance.head(10))
