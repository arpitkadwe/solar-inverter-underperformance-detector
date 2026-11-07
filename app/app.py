"""Solar Inverter Underperformance Detector (Streamlit app)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "HOUR"]
MIN_IRRADIATION = 0.05   # below this it is dawn/dusk/night
MIN_EXPECTED_KW = 50     # ratios are unstable at very low output

st.set_page_config(page_title="Solar Inverter Underperformance Detector",
                   page_icon="☀️", layout="wide")


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "expected_power_xgb.json"))
    return model


@st.cache_data
def load_presets():
    with open(ROOT / "app" / "presets.json") as f:
        return json.load(f)


def expected_power(model, frame):
    pred = np.clip(model.predict(frame[FEATURES]), 0, None)
    return np.where(frame["IRRADIATION"] > MIN_IRRADIATION, pred, 0.0)


model = load_model()
presets = load_presets()

with st.sidebar:
    st.title("Settings")
    threshold = st.slider("Alert threshold (actual ÷ expected)", 0.70, 0.95, 0.85, 0.01)
    st.caption(
        "In simulated-fault tests, 0.85 caught every sustained loss of 20% or more "
        "with about 0.1% false alarms on clean data. Lower thresholds miss smaller faults."
    )

st.title("☀️ Solar Inverter Underperformance Detector")
st.markdown(
    "An XGBoost model estimates how much power a **healthy** inverter should produce "
    "from weather and time of day. Output well below that estimate raises an alert. "
    "An alert means *lower than expected*, not a confirmed hardware fault."
)

tab1, tab2, tab3 = st.tabs(["Single reading", "Batch analysis", "Method & results"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    names = ["Custom"] + [p["name"] for p in presets]
    choice = st.selectbox("Start from an example", names)
    p = next((x for x in presets if x["name"] == choice),
             {"irradiation": 0.7, "ambient": 30.0, "module": 45.0, "hour": 12.0,
              "power": 700.0, "note": ""})
    if p["note"]:
        st.caption(p["note"])

    c1, c2 = st.columns(2)
    with c1:
        irr = st.number_input("Irradiation (kW/m²)", 0.0, 1.5, float(p["irradiation"]), 0.01, key=f"i{choice}")
        amb = st.number_input("Ambient temperature (°C)", 0.0, 60.0, float(p["ambient"]), 0.5, key=f"a{choice}")
        mod = st.number_input("Module temperature (°C)", 0.0, 90.0, float(p["module"]), 0.5, key=f"m{choice}")
        hour = st.slider("Time of day (hour)", 5.0, 20.0, float(p["hour"]), 0.25, key=f"h{choice}")
    with c2:
        actual = st.number_input("Actual AC power (kW)", 0.0, 1600.0, float(p["power"]), 10.0, key=f"p{choice}")

    row = pd.DataFrame([[irr, amb, mod, hour]], columns=FEATURES)
    exp_kw = float(expected_power(model, row)[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Expected power", f"{exp_kw:.0f} kW")
    m2.metric("Actual power", f"{actual:.0f} kW")

    if irr <= MIN_IRRADIATION or exp_kw < MIN_EXPECTED_KW:
        m3.metric("Actual ÷ expected", "n/a")
        st.info("Too little sunlight for a reliable comparison.")
    else:
        ratio = actual / exp_kw
        m3.metric("Actual ÷ expected", f"{ratio:.2f}")
        if ratio < threshold:
            st.error(f"Possible underperformance: output is {100 * (1 - ratio):.0f}% below expected.")
        else:
            st.success("Output is within the normal range for these conditions.")
        st.caption("Single readings are noisier than the 1-hour smoothed check used in batch analysis.")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.markdown("Upload telemetry with columns `DATE_TIME`, `IRRADIATION`, `AMBIENT_TEMPERATURE`, "
                "`MODULE_TEMPERATURE`, `AC_POWER` (and optionally `SOURCE_KEY` for several inverters).")
    up = st.file_uploader("CSV file", type=["csv"])
    use_sample = st.checkbox("Use the included sample file", value=up is None)

    data = None
    if up is not None:
        data = pd.read_csv(up)
    elif use_sample:
        data = pd.read_csv(ROOT / "sample_data" / "sample_telemetry.csv")

    if data is not None:
        need = {"DATE_TIME", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "AC_POWER"}
        if not need.issubset(data.columns):
            st.error(f"Missing columns: {sorted(need - set(data.columns))}")
        else:
            df = data.copy()
            df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
            if "SOURCE_KEY" not in df.columns:
                df["SOURCE_KEY"] = "Inverter"
            df = df.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
            df["HOUR"] = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60
            df["DATE"] = df["DATE_TIME"].dt.date

            df["EXPECTED"] = expected_power(model, df)
            valid = (df["IRRADIATION"] > MIN_IRRADIATION) & (df["EXPECTED"] >= MIN_EXPECTED_KW)
            df["RATIO"] = np.where(valid, df["AC_POWER"] / df["EXPECTED"].clip(lower=1), np.nan)
            df["RATIO_1H"] = (df.groupby(["SOURCE_KEY", "DATE"])["RATIO"]
                                .transform(lambda s: s.rolling(4, min_periods=4).median()))
            df["FLAGGED"] = df["RATIO_1H"] < threshold

            summary = (df.groupby("SOURCE_KEY")
                         .agg(mean_ratio=("RATIO", "mean"), share_flagged=("FLAGGED", "mean"))
                         .round(3).sort_values("mean_ratio"))
            st.subheader("Per-inverter summary")
            st.dataframe(summary, use_container_width=True)

            inv = st.selectbox("Inspect inverter", summary.index)
            d = df[df["SOURCE_KEY"] == inv]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["DATE_TIME"], y=d["EXPECTED"], name="Expected",
                                     line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=d["DATE_TIME"], y=d["AC_POWER"], name="Actual"))
            f = d[d["FLAGGED"]]
            if not f.empty:
                fig.add_trace(go.Scatter(x=f["DATE_TIME"], y=f["AC_POWER"], mode="markers",
                                         name="Flagged", marker=dict(color="red", size=8, symbol="x")))
            fig.update_layout(template="plotly_white", yaxis_title="AC power (kW)",
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

            flagged = df[df["FLAGGED"]][["DATE_TIME", "SOURCE_KEY", "AC_POWER", "EXPECTED", "RATIO_1H"]]
            st.write(f"Flagged readings: {len(flagged)}")
            if len(flagged):
                st.dataframe(flagged.round(2), use_container_width=True)
                st.download_button("Download flagged readings", flagged.to_csv(index=False),
                                   "flagged_readings.csv")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Method")
    st.markdown(
        "- **Data:** public solar plant dataset (22 inverters, 34 days, 15-minute readings).\n"
        "- **Ground truth:** an inverter is labelled underperforming when its 1-hour median output is "
        "below 80% of the median of its peers at the same time.\n"
        "- **Model:** XGBoost trained only on healthy periods to predict expected AC power from "
        "irradiation, ambient and module temperature, and time of day.\n"
        "- **Validation:** chronological split and time-series cross-validation, plus simulated "
        "power-loss faults injected into held-out healthy data."
    )
    res = ROOT / "results" / "model_comparison.csv"
    if res.exists():
        st.subheader("Expected-power model comparison")
        st.dataframe(pd.read_csv(res), use_container_width=True)
    for img, cap in [("detection_curve.png", "Detection rate vs size of simulated fault"),
                     ("peer_ratio_by_inverter.png", "Output relative to peers"),
                     ("shap_summary.png", "What the model relies on (SHAP)")]:
        path = ROOT / "images" / img
        if path.exists():
            st.image(str(path), caption=cap)
    st.subheader("Limitations")
    st.markdown(
        "- One plant and 34 days of data; the test period is about 6 days.\n"
        "- Faults in the evaluation are simulated, not confirmed real failures.\n"
        "- The model uses time of day, so it is specific to this plant and needs retraining elsewhere.\n"
        "- It detects sustained losses of roughly 15-20% or more, not small ones."
    )