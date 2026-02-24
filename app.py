import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pydeck as pdk
import json
import requests
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Urban Logistics AI", layout="wide")
st.title("🚚 Intelligent Urban Logistics Optimization System")
st.caption("Deep Learning-Based Strategic Last-Mile Delivery Decision Support")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_dl_model():
    return load_model("models/dl_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

model = load_dl_model()
scaler = load_scaler()
expected_features = scaler.feature_names_in_

with open("models/model_metrics.json", "r") as f:
    metrics = json.load(f)

feature_importance = pd.read_csv("models/feature_importance.csv")

# --------------------------------------------------
# ORS CONFIG (STREAMLIT SECRETS SAFE CHECK)
# --------------------------------------------------
if "ORS_API_KEY" not in st.secrets:
    st.error("ORS_API_KEY not configured in Streamlit secrets.")
    st.stop()

ORS_API_KEY = st.secrets["ORS_API_KEY"]

@st.cache_data
def get_ors_route(start_lat, start_lon, end_lat, end_lon):

    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

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
        st.error("Routing service unavailable.")
        return None, None, None

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
# GET ROUTE
# --------------------------------------------------
route_geometry, distance, ors_duration = get_ors_route(
    store_lat, store_lon,
    customer_lat, customer_lon
)

if distance is None:
    st.stop()

# --------------------------------------------------
# PREPARE MODEL INPUT
# --------------------------------------------------
input_dict = {}

for feature in expected_features:

    if feature.lower() in ["restaurant_latitude", "store_latitude"]:
        input_dict[feature] = store_lat

    elif feature.lower() in ["restaurant_longitude", "store_longitude"]:
        input_dict[feature] = store_lon

    elif feature.lower() in ["delivery_location_latitude", "customer_latitude"]:
        input_dict[feature] = customer_lat

    elif feature.lower() in ["delivery_location_longitude", "customer_longitude"]:
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

# Strategic Layer
sla_threshold = 40

if predicted_time > sla_threshold:
    st.error("⚠ High Delay Risk - Increase manpower or prioritize order")
elif predicted_time > 30:
    st.warning("⚠ Moderate Delay Risk - Monitor preparation closely")
else:
    st.success("✔ Low Delay Risk - Operations Stable")

improvement = ((predicted_time - optimized_time) / predicted_time) * 100
st.write(f"📈 Estimated Efficiency Gain: {improvement:.2f}%")

# --------------------------------------------------
# MAP
# --------------------------------------------------
st.subheader("🗺 Real-Time Delivery Route")

route_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": route_geometry}],
    get_path="path",
    get_color=[255, 0, 0],
    width_min_pixels=5,
)

marker_data = pd.DataFrame({
    "lat": [store_lat, customer_lat],
    "lon": [store_lon, customer_lon],
})

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=marker_data,
    get_position='[lon, lat]',
    get_radius=150,
    get_fill_color=[0, 255, 0],
)

view_state = pdk.ViewState(
    latitude=(store_lat + customer_lat) / 2,
    longitude=(store_lon + customer_lon) / 2,
    zoom=12,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[route_layer, scatter_layer],
        initial_view_state=view_state,
        map_style="road",
    )
)

# --------------------------------------------------
# ADVANCED MODEL ANALYSIS
# --------------------------------------------------
with st.expander("📊 Advanced Model Evaluation"):

    st.write("Random Forest MAE:", round(metrics["rf_mae"], 2))
    st.write("Random Forest R²:", round(metrics["rf_r2"], 2))
    st.write("Deep Learning MAE:", round(metrics["dl_mae"], 2))
    st.write("Deep Learning R²:", round(metrics["dl_r2"], 2))

    st.markdown("### Top Feature Importance")
    st.dataframe(feature_importance.head(10))
