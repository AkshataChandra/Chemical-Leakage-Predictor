import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Chemical Leakage Risk Dashboard", layout="wide")

st.title("🧪 Chemical Leakage Dynamic Risk Dashboard")

# -----------------------------------------------------
# Load model and preprocessing artifacts
# -----------------------------------------------------
model  = joblib.load("leakagemodel_calibrated.pkl")
scaler = joblib.load("scaler.pkl")
features = [str(f) for f in joblib.load("selected_features.pkl")]
calib_df = pd.read_csv("calibration_data.csv")

# -----------------------------------------------------
# Feature preparation
# -----------------------------------------------------
def prepare_features(d):

    if isinstance(d, dict):
        df = pd.DataFrame([d])
    else:
        df = d.copy()

    df["volume_diff"] = df["initialvolume"] - df["transferredvolume"]
    df["percent_loss"] = (df["volume_diff"] / (df["initialvolume"] + 1e-9)) * 100
    df["temp_diff"] = df["ambienttemp"] - df["chemicalevappoint"]
    df["evaporation_ratio"] = df["ambienttemp"] / (df["chemicalevappoint"] + 1e-9)
    df["safety_margin"] = (df["chemicalevappoint"] - df["ambienttemp"]).clip(lower=0)

    T = df["ambienttemp"] + 273.15
    bp = df["chemicalevappoint"] + 273.15
    mw = df.get("chemical_mw", pd.Series([100] * len(df)))

    df["vapor_pressure_proxy"] = np.exp(-mw / 500.0) * np.exp((T - bp) / (50.0 + 0.01 * np.abs(bp)))

    if "chemical_mw" not in df.columns:
        df["chemical_mw"] = 100.0

    X = df[features]
    X_scaled = scaler.transform(X)

    return X_scaled, df


# -----------------------------------------------------
# Safety recommendations
# -----------------------------------------------------
def safety_recommendations(prob, df):

    recs = []

    if prob >= 0.85:
        recs.append("🚨 Immediate action: stop transfer and contain leak.")
    elif prob >= 0.6:
        recs.append("⚠ High risk: inspect seals and pressure control.")
    elif prob >= 0.4:
        recs.append("🟡 Moderate risk: monitor and cool system.")
    else:
        recs.append("🟢 Low risk: normal monitoring.")

    if df["ambienttemp"].iloc[0] > df["chemicalevappoint"].iloc[0]:
        recs.append("Ambient temperature above evaporation point — cool system.")

    if df["pressure"].iloc[0] < 0.9:
        recs.append("Low pressure increases evaporation risk.")

    if df["percent_loss"].iloc[0] > 5:
        recs.append("Percent loss >5% — inspect transfer lines.")

    return recs


# -----------------------------------------------------
# Visualization functions
# -----------------------------------------------------
def plot_heatmap(center_temp, center_humidity, df_params):

    temps = np.linspace(center_temp - 40, center_temp + 40, 60)
    hums  = np.linspace(max(5, center_humidity - 40), min(95, center_humidity + 40), 60)

    grid = []

    for T in temps:
        for H in hums:
            g = df_params.copy()
            g["ambienttemp"] = T
            g["humidity"] = H
            grid.append(g)

    grid_df = pd.DataFrame(grid)

    Xs, _ = prepare_features(grid_df)

    p = model.predict_proba(Xs)[:,1]

    Z = p.reshape(len(temps), len(hums))

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=hums,
        y=temps,
        colorscale="RdYlGn_r"
    ))

    fig.update_layout(
        title="Dynamic Risk Heatmap",
        xaxis_title="Humidity (%)",
        yaxis_title="Temperature (°C)"
    )

    return fig


def plot_3d(center_temp, center_pressure, df_params):

    temps = np.linspace(center_temp - 40, center_temp + 40, 40)
    press = np.linspace(max(0.7, center_pressure - 0.4), min(1.5, center_pressure + 0.4), 40)

    grid = []

    for T in temps:
        for P in press:
            g = df_params.copy()
            g["ambienttemp"] = T
            g["pressure"] = P
            grid.append(g)

    grid_df = pd.DataFrame(grid)

    Xs, _ = prepare_features(grid_df)

    probs = model.predict_proba(Xs)[:,1].reshape(len(temps), len(press))

    fig = go.Figure(go.Surface(
        x=press,
        y=temps,
        z=probs,
        colorscale="RdYlGn_r"
    ))

    fig.update_layout(
        title="3D Dynamic Risk Surface",
        scene=dict(
            xaxis_title="Pressure (atm)",
            yaxis_title="Temperature (°C)",
            zaxis_title="Leak Probability"
        )
    )

    return fig


def plot_calibration():

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=calib_df["prob_pred"],
        y=calib_df["prob_true"],
        mode="lines+markers",
        name="Calibrated"
    ))

    fig.add_trace(go.Scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        line=dict(dash="dash"),
        name="Perfect"
    ))

    fig.update_layout(
        title="Model Calibration Curve",
        xaxis_title="Predicted",
        yaxis_title="Observed"
    )

    return fig


# -----------------------------------------------------
# Sidebar Inputs
# -----------------------------------------------------
st.sidebar.header("Input Parameters")

initialvolume = st.sidebar.number_input("Initial Volume", value=1000.0)
transferredvolume = st.sidebar.number_input("Transferred Volume", value=800.0)
ambienttemp = st.sidebar.number_input("Ambient Temperature (°C)", value=25.0)
chemicalevappoint = st.sidebar.number_input("Chemical Evaporation Point", value=78.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
pressure = st.sidebar.number_input("Pressure (atm)", value=1.0)
chemical_mw = st.sidebar.number_input("Chemical Molecular Weight", value=100.0)

input_data = {
    "initialvolume": initialvolume,
    "transferredvolume": transferredvolume,
    "ambienttemp": ambienttemp,
    "chemicalevappoint": chemicalevappoint,
    "humidity": humidity,
    "pressure": pressure,
    "chemical_mw": chemical_mw
}

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
X_scaled, df_row = prepare_features(input_data)

prob = float(model.predict_proba(X_scaled)[0][1])

st.subheader("Leakage Probability")

st.metric("Leak Probability", f"{prob:.2f}")

# -----------------------------------------------------
# Recommendations
# -----------------------------------------------------
recs = safety_recommendations(prob, df_row)

st.subheader("Safety Recommendations")

for r in recs:
    st.write(r)

# -----------------------------------------------------
# Charts
# -----------------------------------------------------
st.subheader("Dynamic Risk Heatmap")
st.plotly_chart(plot_heatmap(ambienttemp, humidity, input_data), use_container_width=True)

st.subheader("3D Risk Surface")
st.plotly_chart(plot_3d(ambienttemp, pressure, input_data), use_container_width=True)

st.subheader("Calibration Curve")
st.plotly_chart(plot_calibration(), use_container_width=True)

