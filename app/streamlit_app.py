import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Customer Risk Intelligence Platform",
    layout="wide"
)

# ==============================
# STYLING
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0B0F17;
    color: #E5E7EB;
}

h1, h2, h3 {
    color: #FFFFFF;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1F2937;
}

.metric-box {
    background: #111827;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #1F2937;
    text-align: center;
}

.approve {color: #10B981; font-weight: 600;}
.review {color: #EF4444; font-weight: 600;}
.conditional {color: #F59E0B; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("models/risk_model.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, feature_cols

model, feature_columns = load_model()

# ==============================
# SESSION STATE
# ==============================
if "step" not in st.session_state:
    st.session_state.step = 1

if "data" not in st.session_state:
    st.session_state.data = None

if "results" not in st.session_state:
    st.session_state.results = None

if "config" not in st.session_state:
    st.session_state.config = {"fraud_cost": 3.0, "review_cost": 4.0}

# ==============================
# HELPERS
# ==============================
def explain(row):
    reasons = []
    if row["order_value"] > 100:
        reasons.append("High transaction value")
    if row["risk_probability"] > 0.7:
        reasons.append("High risk score")
    return ", ".join(reasons[:2]) if reasons else "Low risk profile"

def map_strategy(s):
    return {
        "AI Automation": "Approve",
        "Human Review": "Review",
        "Hybrid": "Conditional"
    }[s]

def cost_ai(p, amt, fc):
    return p * amt * fc * 0.3

def cost_human(p, amt, fc, rc):
    return rc + (p * amt * fc * 0.1)

def cost_hybrid(p, amt, fc, rc):
    return cost_ai(p, amt, fc) if p < 0.4 else cost_human(p, amt, fc, rc)

def choose(row):
    costs = {
        "AI Automation": row["cost_ai"],
        "Human Review": row["cost_human"],
        "Hybrid": row["cost_hybrid"]
    }
    return min(costs, key=costs.get)

# ==============================
# HEADER
# ==============================
st.title("Customer Risk Intelligence Platform")
st.caption("Optimize transaction decisions by minimizing financial loss")

# ==============================
# STEP PROGRESS
# ==============================
steps = ["Upload Data", "Map Columns", "Set Costs", "Generate Decisions", "Results"]
st.write(f"Step {st.session_state.step} of 5: {steps[st.session_state.step - 1]}")

st.divider()

# ==============================
# STEP 1: UPLOAD
# ==============================
if st.session_state.step == 1:

    st.subheader("Upload Transaction Data")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.session_state.raw = df
        st.success("Data uploaded successfully")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Continue"):
            st.session_state.step = 2

# ==============================
# STEP 2: MAPPING
# ==============================
elif st.session_state.step == 2:

    df = st.session_state.raw

    st.subheader("Map Your Data")

    mapping = {}
    cols = df.columns

    for col in feature_columns + ["order_value"]:
        mapping[col] = st.selectbox(f"{col}", cols)

    if st.button("Confirm Mapping"):
        df = df.rename(columns={v: k for k, v in mapping.items()})
        df = df[feature_columns + ["order_value"]]
        df = df.fillna(0)

        st.session_state.data = df
        st.success("Data prepared successfully")
        st.session_state.step = 3

# ==============================
# STEP 3: COSTS
# ==============================
elif st.session_state.step == 3:

    st.subheader("Configure Business Costs")

    fc = st.slider("Fraud Loss Multiplier", 1.0, 5.0, 3.0)
    rc = st.slider("Manual Review Cost", 1.0, 20.0, 4.0)

    st.session_state.config = {"fraud_cost": fc, "review_cost": rc}

    if st.button("Continue"):
        st.session_state.step = 4

# ==============================
# STEP 4: RUN
# ==============================
elif st.session_state.step == 4:

    st.subheader("Generate Decisions")

    if st.button("Run Decision Engine"):

        with st.spinner("Analyzing transactions..."):

            df = st.session_state.data.copy()
            cfg = st.session_state.config

            X = df[feature_columns]
            df["risk_probability"] = model.predict_proba(X)[:, 1]

            df["cost_ai"] = df.apply(lambda x: cost_ai(x["risk_probability"], x["order_value"], cfg["fraud_cost"]), axis=1)
            df["cost_human"] = df.apply(lambda x: cost_human(x["risk_probability"], x["order_value"], cfg["fraud_cost"], cfg["review_cost"]), axis=1)
            df["cost_hybrid"] = df.apply(lambda x: cost_hybrid(x["risk_probability"], x["order_value"], cfg["fraud_cost"], cfg["review_cost"]), axis=1)

            df["optimal_strategy"] = df.apply(choose, axis=1)
            df["expected_cost"] = df[["cost_ai","cost_human","cost_hybrid"]].min(axis=1)

            df["decision"] = df["optimal_strategy"].apply(map_strategy)
            df["reason"] = df.apply(explain, axis=1)

            st.session_state.results = df

        st.success("Decisions generated successfully")
        st.session_state.step = 5

# ==============================
# STEP 5: RESULTS
# ==============================
elif st.session_state.step == 5:

    df = st.session_state.results

    st.subheader("Business Impact")

    baseline = df["cost_human"].sum()
    optimized = df["expected_cost"].sum()
    savings = baseline - optimized
    reduction = (savings / baseline) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Cost", f"${baseline:,.0f}")
    col2.metric("Optimized Cost", f"${optimized:,.0f}")
    col3.metric("Savings", f"${savings:,.0f}", f"{reduction:.1f}%")

    st.info(f"You saved ${savings:,.0f} using optimized decisions")

    st.divider()

    st.subheader("Decision Table")

    view = df[[
        "decision",
        "reason",
        "risk_probability",
        "expected_cost"
    ]].rename(columns={
        "decision": "Recommended Action",
        "reason": "Reason",
        "risk_probability": "Fraud Risk Score",
        "expected_cost": "Estimated Loss"
    })

    st.dataframe(view.sort_values("Fraud Risk Score", ascending=False), use_container_width=True)

    st.divider()

    st.subheader("Strategy Comparison")

    total_ai = df["cost_ai"].sum()
    total_human = df["cost_human"].sum()
    total_opt = df["expected_cost"].sum()

    comp = pd.DataFrame({
        "Strategy": ["AI Only", "Human Only", "Optimized"],
        "Total Cost": [total_ai, total_human, total_opt]
    })

    st.dataframe(comp, use_container_width=True)

    st.divider()

    st.subheader("Adjust Scenario")

    fc = st.slider("Fraud Cost Scenario", 1.0, 5.0, st.session_state.config["fraud_cost"])
    rc = st.slider("Review Cost Scenario", 1.0, 20.0, st.session_state.config["review_cost"])

    st.caption("Changing costs shifts decision behavior dynamically")
