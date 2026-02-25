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
# LOAD MODELS & FILES (Paths fixed for root directory)
# --------------------------------------------------
@st.cache_resource
def load_assets():
    # Files are in the root folder as per your GitHub screenshot
    model = load_model("dl_model.keras") 
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
    expected_features = scaler.feature_names_in_

    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)

    feature_importance = pd.read_csv("feature_importance.csv")
except Exception as e:
    st.error(f"Error loading model assets: {e}. Ensure all files are in the main GitHub folder.")
    st.stop()

# --------------------------------------------------
# ORS CONFIG (Fixed 403 Forbidden Error)
# --------------------------------------------------
ORS_API_KEY = st.secrets["ORS_API_KEY"]

@st.cache_data
def get_ors_route(start_lat, start_lon, end_lat, end_lon):
    # Passing key in URL to bypass some 403 Header restrictions
    url = f"https://api.openrouteservice.org/v2/directions/driving-car?api_key={ORS_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    body = {
        "coordinates": [
            [start_lon, start_lat],
            [end_lon, end_lat]
        ]
    }

    try:
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        geometry = data["features"][0]["geometry"]["coordinates"]
        distance = data["features"][0]["properties"]["summary"]["distance"] / 1000
        duration = data["features"][0]["properties"]["summary"]["duration"] / 60
        return geometry, distance, duration
    except Exception as e:
        st.error(f"Routing Error: {e}")
        return None, None, None

# --------------------------------------------------
# SIDEBAR INPUTS - Strategic Parameters
# --------------------------------------------------
st.sidebar.header("📍 Delivery Details")
store_lat = st.sidebar.number_input("Store Latitude", value=19.2089)
store_lon = st.sidebar.number_input("Store Longitude", value=72.8722)
customer_lat = st.sidebar.number_input("Customer Latitude", value=19.2108)
customer_lon = st.sidebar.number_input("Customer Longitude", value=72.8746)

st.sidebar.markdown("---")
st.sidebar.header("☁️ Operational Environment")
traffic_level = st.sidebar.selectbox("🚦 Traffic Level", ["Low", "Moderate", "High"])
weather_condition = st.sidebar.selectbox("🌦 Weather Condition", ["Clear", "Cloudy", "Foggy", "Sandstorms", "Windy", "Stormy"])
is_festival = st.sidebar.checkbox("🎉 Festival/Holiday Period")

st.sidebar.markdown("---")
store_rating = st.sidebar.slider("Store Rating", 1.0, 5.0, 4.0)
order_cost = st.sidebar.number_input("Order Cost (₹)", value=300)

# --------------------------------------------------
# GEOSPATIAL & AI LOGIC
# --------------------------------------------------
route_geometry, distance, ors_duration = get_ors_route(store_lat, store_lon, customer_lat, customer_lon)

if distance is None:
    st.info("Please verify your ORS API Key and coordinates.")
    st.stop()

# Build Model Input
input_dict = {feat: 0 for feat in expected_features}
for feature in expected_features:
    f_lower = feature.lower()
    if any(x in f_lower for x in ["restaurant_latitude", "store_latitude"]): input_dict[feature] = store_lat
    elif any(x in f_lower for x in ["restaurant_longitude", "store_longitude"]): input_dict[feature] = store_lon
    elif any(x in f_lower for x in ["delivery_location_latitude", "customer_latitude"]): input_dict[feature] = customer_lat
    elif any(x in f_lower for x in ["delivery_location_longitude", "customer_longitude"]): input_dict[feature] = customer_lon
    elif "distance" in f_lower: input_dict[feature] = distance
    elif "rating" in f_lower: input_dict[feature] = store_rating
    elif "cost" in f_lower: input_dict[feature] = order_cost

input_df = pd.DataFrame([input_dict])[expected_features]
input_scaled = scaler.transform(input_df)

# Deep Learning Prediction
model_time = model.predict(input_scaled, verbose=0)[0][0]
prep_time = 10 
logic_time = prep_time + ors_duration

# Weighted base prediction
base_predicted_time = (model_time * 0.6) + (logic_time * 0.4)

# STRATEGIC MULTIPLIERS (As per Project Objectives)
traffic_factor = {"Low": 1.0, "Moderate": 1.2, "High": 1.5}[traffic_level]
weather_factor = {"Clear": 1.0, "Cloudy": 1.1, "Foggy": 1.3, "Sandstorms": 1.4, "Windy": 1.2, "Stormy": 1.6}[weather_condition]
festival_multiplier = 1.3 if is_festival else 1.0

# Final Calculations
predicted_time = base_predicted_time * weather_factor * festival_multiplier
optimized_time = predicted_time / traffic_factor

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
st.subheader("📊 Operational Prediction Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("📏 Road Distance", f"{distance:.2f} km")
col2.metric("⏱ Predicted Time", f"{predicted_time:.2f} mins")
col3.metric("🚀 Optimized Target", f"{optimized_time:.2f} mins")

# Strategic Risk Assessment Layer
sla_threshold = 40
if predicted_time > sla_threshold:
    st.error(f"⚠️ **High Delay Risk** ({weather_condition} conditions + Traffic). Strategic intervention required.")
elif predicted_time > 30:
    st.warning("⚠️ **Moderate Delay Risk** - Monitor courier preparation.")
else:
    st.success("✅ **Low Delay Risk** - Operations Stable.")

improvement = ((predicted_time - optimized_time) / predicted_time) * 100
st.info(f"📈 **Strategic Insight:** Smart scheduling could reduce delivery time by **{improvement:.2f}%**.")

# --------------------------------------------------
# MAP & ANALYSIS
# --------------------------------------------------
st.subheader("🗺 Real-Time Delivery Route")
view_state = pdk.ViewState(latitude=(store_lat + customer_lat) / 2, longitude=(store_lon + customer_lon) / 2, zoom=13)
route_layer = pdk.Layer("PathLayer", data=[{"path": route_geometry}], get_path="path", get_color=[255, 0, 0], width_min_pixels=5)
scatter_layer = pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat": [store_lat, customer_lat], "lon": [store_lon, customer_lon]}), get_position='[lon, lat]', get_radius=100, get_fill_color=[0, 255, 0])

st.pydeck_chart(pdk.Deck(layers=[route_layer, scatter_layer], initial_view_state=view_state, map_style="road"))

with st.expander("📊 Advanced Model Evaluation & Feature Importance"):
    c1, c2 = st.columns(2)
    c1.write(f"**Random Forest R²:** {metrics['rf_r2']:.2f}")
    c2.write(f"**Deep Learning R²:** {metrics['dl_r2']:.2f}")
    st.bar_chart(feature_importance.set_index("feature").head(10))
