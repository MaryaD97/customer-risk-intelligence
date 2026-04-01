import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Fraud Decision Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# GLOBAL STYLING (PREMIUM UI)
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0B0F17;
    color: #E5E7EB;
}

h1, h2, h3 {
    color: #FFFFFF;
    letter-spacing: -0.3px;
}

section[data-testid="stSidebar"] {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Cards */
.card {
    background: linear-gradient(145deg, #111827, #0B0F17);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1F2937;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Buttons */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 18px;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 15px;
    border-radius: 10px;
}

/* Divider */
hr {
    border: 1px solid #1F2937;
}
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
if "mapped_data" not in st.session_state:
    st.session_state.mapped_data = None

if "results" not in st.session_state:
    st.session_state.results = None

if "config" not in st.session_state:
    st.session_state.config = {"fraud_cost": 3.0, "review_cost": 4.0}

# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.title("Fraud Decision Engine")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "1. Upload Data",
        "2. Set Costs",
        "3. Run Analysis",
        "4. Decisions",
        "5. Insights"
    ]
)

# ==============================
# HELPER FUNCTIONS
# ==============================
def risk_tier(p):
    return "Low" if p < 0.3 else "Medium" if p < 0.7 else "High"

def cost_ai(p, amt, fraud_cost):
    return p * amt * fraud_cost * 0.3

def cost_human(p, amt, fraud_cost, review_cost):
    return review_cost + (p * amt * fraud_cost * 0.1)

def cost_hybrid(p, amt, fraud_cost, review_cost):
    return cost_ai(p, amt, fraud_cost) if p < 0.4 else cost_human(p, amt, fraud_cost, review_cost)

def choose_strategy(row):
    costs = {
        "AI Automation": row["cost_ai"],
        "Human Review": row["cost_human"],
        "Hybrid": row["cost_hybrid"]
    }
    return min(costs, key=costs.get)

import shap

@st.cache_resource
def load_explainer():
    explainer = shap.Explainer(model)
    return explainer

explainer = load_explainer()

def simulate_decisions(df, fraud_cost, review_cost):
    df = df.copy()

    df["cost_ai"] = df.apply(lambda x: cost_ai(x["risk_probability"], x["order_value"], fraud_cost), axis=1)
    df["cost_human"] = df.apply(lambda x: cost_human(x["risk_probability"], x["order_value"], fraud_cost, review_cost), axis=1)
    df["cost_hybrid"] = df.apply(lambda x: cost_hybrid(x["risk_probability"], x["order_value"], fraud_cost, review_cost), axis=1)

    df["optimal_strategy"] = df.apply(choose_strategy, axis=1)
    df["expected_cost"] = df[["cost_ai", "cost_human", "cost_hybrid"]].min(axis=1)

    return df


def estimate_baseline_cost(df):
    # naive baseline: always human review
    return df["cost_human"].sum()
    

# ==============================
# OVERVIEW PAGE
# ==============================
if page == "Overview":

    st.title("Fraud Decision Engine")

    st.markdown("""
    Automatically choose the **lowest-cost action** for every transaction by combining fraud prediction with financial impact modeling.
    """)

    st.divider()

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Loss Reduction", "Optimized")
    col2.metric("Manual Reviews", "Minimized")
    col3.metric("Decision Speed", "Instant")

    st.divider()

    # FLOW
    st.subheader("How It Works")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("**Upload Data**  \nTransaction-level dataset")
    c2.markdown("**Risk Scoring**  \nPredict fraud probability")
    c3.markdown("**Cost Simulation**  \nEvaluate decision cost")
    c4.markdown("**Decision Output**  \nSelect optimal action")

    st.divider()

    # OUTPUT
    st.subheader("What You Get")

    st.markdown("""
    - Fraud probability for each transaction  
    - Recommended action (AI / Human / Hybrid)  
    - Expected cost per decision  
    - Explanation of key drivers  
    """)

    st.divider()

    # CTA
    st.subheader("Get Started")

    col1, col2, col3 = st.columns(3)

    col1.markdown("**1. Upload Data**")
    col2.markdown("**2. Set Costs**")
    col3.markdown("**3. Run Analysis**")

    if st.session_state.results is None:
        st.info("No analysis yet. Start by uploading your data.")

# ==============================
# UPLOAD
# ==============================
elif page == "1. Upload Data":

    st.title("Upload Transaction Data")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Map Columns")

        mapping = {}
        for col in feature_columns + ["order_value"]:
            mapping[col] = st.selectbox(col, df.columns)

        if st.button("Confirm Mapping"):
            df = df.rename(columns={v: k for k, v in mapping.items()})
            st.session_state.mapped_data = df
            st.success("Data ready")

# ==============================
# CONFIG
# ==============================
elif page == "2. Set Costs":

    st.title("Business Cost Configuration")

    col1, col2 = st.columns(2)

    fraud_cost = col1.slider("Fraud Loss Multiplier", 1.0, 5.0, 3.0)
    review_cost = col2.slider("Manual Review Cost", 1.0, 20.0, 4.0)

    st.session_state.config = {
        "fraud_cost": fraud_cost,
        "review_cost": review_cost
    }

# ==============================
# RUN
# ==============================
elif page == "3. Run Analysis":

    st.title("Run Decision Engine")

    if st.session_state.mapped_data is None:
        st.warning("Upload data first")
    else:
        if st.button("Run Analysis"):

            df = st.session_state.mapped_data.copy()
            cfg = st.session_state.config

            X = df[feature_columns]
            df["risk_probability"] = model.predict_proba(X)[:, 1]

            df["cost_ai"] = df.apply(lambda x: cost_ai(x["risk_probability"], x["order_value"], cfg["fraud_cost"]), axis=1)
            df["cost_human"] = df.apply(lambda x: cost_human(x["risk_probability"], x["order_value"], cfg["fraud_cost"], cfg["review_cost"]), axis=1)
            df["cost_hybrid"] = df.apply(lambda x: cost_hybrid(x["risk_probability"], x["order_value"], cfg["fraud_cost"], cfg["review_cost"]), axis=1)

            df["optimal_strategy"] = df.apply(choose_strategy, axis=1)
            df["expected_cost"] = df[["cost_ai", "cost_human", "cost_hybrid"]].min(axis=1)
            df["risk_tier"] = df["risk_probability"].apply(risk_tier)

            st.session_state.results = df

            st.success("Analysis complete")

# ==============================
# DECISIONS
# ==============================
elif page == "4. Decisions":

    st.title("Decision Simulator")

    if st.session_state.results is None:
        st.warning("Run analysis first")
    else:
        base_df = st.session_state.results

        st.subheader("Adjust Business Costs")

        col1, col2 = st.columns(2)

        sim_fraud = col1.slider("Fraud Cost", 1.0, 5.0, st.session_state.config["fraud_cost"])
        sim_review = col2.slider("Review Cost", 1.0, 20.0, st.session_state.config["review_cost"])

        sim_df = simulate_decisions(base_df, sim_fraud, sim_review)

        st.divider()

        # KPIs
        total_cost = sim_df["expected_cost"].sum()
        automation_rate = (sim_df["optimal_strategy"].str.contains("AI")).mean()
        high_risk = (sim_df["risk_probability"] > 0.7).mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost", f"${total_cost:,.0f}")
        col2.metric("Automation Rate", f"{automation_rate:.1%}")
        col3.metric("High Risk Transactions", f"{high_risk:.1%}")

        st.divider()

        st.dataframe(sim_df[[
            "risk_probability",
            "risk_tier",
            "optimal_strategy",
            "expected_cost"
        ]], use_container_width=True)

        st.divider()

        st.subheader("Explain a Decision")

        selected_index = st.selectbox("Select Transaction", sim_df.index)

        row = sim_df.loc[selected_index]

        st.markdown(f"""
        **Prediction Details**
        - Risk Probability: {row['risk_probability']:.2f}
        - Selected Strategy: {row['optimal_strategy']}
        - Expected Cost: ${row['expected_cost']:.2f}
        """)
        
        # SHAP Explanation
        X_row = base_df.loc[[selected_index], feature_columns]
        shap_values = explainer(X_row)
        
        st.write("Feature Impact (SHAP)")
        st.bar_chart(pd.DataFrame({
            "feature": feature_columns,
            "impact": shap_values.values[0]
        }).set_index("feature"))

# ==============================
# INSIGHTS
# ==============================
elif page == "5. Insights":

    st.title("Executive Dashboard")

    if st.session_state.results is None:
        st.warning("Run analysis first")
    else:
        df = st.session_state.results

        baseline = estimate_baseline_cost(df)
        optimized = df["expected_cost"].sum()
        savings = baseline - optimized

        st.subheader("Business Impact")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Cost", f"${baseline:,.0f}")
        col2.metric("Optimized Cost", f"${optimized:,.0f}")
        col3.metric("Estimated Savings", f"${savings:,.0f}")

        st.divider()

        st.subheader("Decision Breakdown")
        st.bar_chart(df["optimal_strategy"].value_counts())

        st.subheader("Risk Distribution")
        st.bar_chart(df["risk_tier"].value_counts())
