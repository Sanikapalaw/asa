import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Strategic Last-Mile Delivery DSS", layout="wide")

st.title("Strategic Last-Mile Delivery Decision Support System")
st.caption("AI-Driven Predictive Analytics and Operational Optimization Framework")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_dl_model():
    if not os.path.exists("dl_model.keras"):
        st.error("Model file not found.")
        st.stop()
    return load_model("dl_model.keras")

@st.cache_resource
def load_scaler():
    if not os.path.exists("scaler.pkl"):
        st.error("Scaler file not found.")
        st.stop()
    return joblib.load("scaler.pkl")

model = load_dl_model()
scaler = load_scaler()
expected_features = scaler.feature_names_in_

# --------------------------------------------------
# LOAD METRICS
# --------------------------------------------------
try:
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
except:
    metrics = {}

try:
    feature_importance = pd.read_csv("feature_importance.csv")
except:
    feature_importance = pd.DataFrame()

# --------------------------------------------------
# SIDEBAR INPUT PARAMETERS
# --------------------------------------------------
st.sidebar.header("Operational Input Parameters")

store_lat = st.sidebar.number_input("Store Latitude", value=19.2089)
store_lon = st.sidebar.number_input("Store Longitude", value=72.8722)

customer_lat = st.sidebar.number_input("Customer Latitude", value=19.2108)
customer_lon = st.sidebar.number_input("Customer Longitude", value=72.8746)

store_rating = st.sidebar.slider("Store Performance Rating", 1.0, 5.0, 4.0)
order_cost = st.sidebar.number_input("Order Value (INR)", value=300)

traffic_level = st.sidebar.selectbox(
    "Traffic Intensity Level",
    ["Low", "Moderate", "High"]
)

# --------------------------------------------------
# DISTANCE CALCULATION
# --------------------------------------------------
distance = np.sqrt(
    (store_lat - customer_lat)**2 +
    (store_lon - customer_lon)**2
) * 111

estimated_travel_time = distance * 4

# --------------------------------------------------
# MODEL INPUT PREPARATION
# --------------------------------------------------
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

input_df = pd.DataFrame([input_dict])[expected_features]
input_scaled = scaler.transform(input_df)

# --------------------------------------------------
# PREDICTION SYSTEM
# --------------------------------------------------

# ML Base Prediction
model_time = model.predict(input_scaled, verbose=0)[0][0]

prep_time = 10
logic_time = prep_time + estimated_travel_time

base_predicted_time = (model_time * 0.6) + (logic_time * 0.4)

# Business Logic Layer
rating_penalty = (5 - store_rating) * 2

cost_penalty = 0
if order_cost > 1000:
    cost_penalty = 3
elif order_cost > 500:
    cost_penalty = 1.5

business_adjusted_time = base_predicted_time + rating_penalty + cost_penalty

# Traffic Multiplier
traffic_factor = {"Low":1.0, "Moderate":1.2, "High":1.5}[traffic_level]
final_operational_time = business_adjusted_time * traffic_factor

# --------------------------------------------------
# DASHBOARD SECTION
# --------------------------------------------------
st.subheader("Delivery Time Forecasting and Risk Assessment")

col1, col2, col3 = st.columns(3)

col1.metric("Estimated Route Distance (km)", f"{distance:.2f}")
col2.metric("Business-Adjusted Delivery Time (mins)", f"{business_adjusted_time:.2f}")
col3.metric("Final Operational Delivery Time (mins)", f"{final_operational_time:.2f}")

st.write("ML Base Prediction (Before Operational Adjustments):", round(base_predicted_time, 2))

# SLA Risk Assessment
sla_threshold = 40

if final_operational_time > sla_threshold:
    st.error("High Delay Risk – Operational intervention recommended.")
elif final_operational_time > 30:
    st.warning("Moderate Delay Risk – Monitor preparation and dispatch.")
else:
    st.success("Low Delay Risk – Operations within acceptable limits.")

impact = ((final_operational_time - base_predicted_time) / base_predicted_time) * 100
st.metric("Operational Impact Increase (%)", f"{impact:.2f}%")

# --------------------------------------------------
# GEOSPATIAL VISUALIZATION
# --------------------------------------------------
st.subheader("Geospatial Route Visualization")

m = folium.Map(
    location=[store_lat, store_lon],
    zoom_start=12,
    tiles="OpenStreetMap"
)

folium.Marker(
    [store_lat, store_lon],
    popup="Store Location",
    icon=folium.Icon(color="green")
).add_to(m)

folium.Marker(
    [customer_lat, customer_lon],
    popup="Customer Location",
    icon=folium.Icon(color="black")
).add_to(m)

folium.PolyLine(
    locations=[[store_lat, store_lon], [customer_lat, customer_lon]],
    color="blue",
    weight=5
).add_to(m)

st_folium(m, width=950, height=500)

# --------------------------------------------------
# MODEL PERFORMANCE SECTION
# --------------------------------------------------
with st.expander("Model Performance Metrics and Feature Analysis"):
    if metrics:
        st.write("Deep Learning MAE:", round(metrics.get("dl_mae", 0), 2))
        st.write("Deep Learning R²:", round(metrics.get("dl_r2", 0), 2))

    if not feature_importance.empty:
        st.markdown("Top Feature Importance")
        st.dataframe(feature_importance.head(10))
