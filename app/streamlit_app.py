import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Customer Risk Intelligence", layout="wide")

# ==============================
# DARK UI STYLING
# ==============================
st.markdown("""
<style>
.main { background-color: #0E1117; color: #FAFAFA; }
h1, h2, h3 { color: #FFFFFF; }

[data-testid="metric-container"] {
    background-color: #1C1F26;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #2A2E39;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
}

hr { border: 1px solid #2A2E39; }
</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.title("Customer Risk Intelligence Platform")

st.markdown("""
Optimize fraud handling decisions using cost-based machine learning.

Upload transaction data, configure business costs, and receive optimized
decisions on whether to automate, review, or escalate transactions.
""")

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
if "data" not in st.session_state:
    st.session_state.data = None

if "mapped_data" not in st.session_state:
    st.session_state.mapped_data = None

if "results" not in st.session_state:
    st.session_state.results = None

if "config" not in st.session_state:
    st.session_state.config = {"fraud_cost": 3.0, "review_cost": 4.0}

# ==============================
# NAVIGATION
# ==============================
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Upload & Map Data", "Configure", "Run Analysis", "Decisions", "Insights"]
)

# ==============================
# HELPER FUNCTIONS
# ==============================
def risk_tier(p):
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"

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

def style_decisions(df):
    def color(val):
        if "AI" in val:
            return "color: #22c55e; font-weight: 600"
        elif "Human" in val:
            return "color: #ef4444; font-weight: 600"
        else:
            return "color: #f59e0b; font-weight: 600"
    return df.style.applymap(color, subset=["optimal_strategy"])

# ==============================
# OVERVIEW
# ==============================
if page == "Overview":

    st.header("Decision Intelligence Overview")

    if st.session_state.results is None:
        st.warning("System status: Awaiting data upload and analysis")
    else:
        st.success("System status: Analysis complete")

    st.subheader("How to Use This Tool")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        1. Upload your dataset  
        2. Map your columns  
        3. Configure cost parameters  
        4. Run analysis  
        """)

    with col2:
        st.markdown("""
        5. Review decisions  
        6. Analyze cost impact  
        7. Export results  
        """)

    st.divider()

    st.subheader("What the System Does")

    st.markdown("""
    Predicts fraud risk and assigns the most cost-efficient handling strategy:

    - AI Automation → low-risk  
    - Human Review → high-risk  
    - Hybrid → medium-risk  

    All decisions minimize expected financial cost.
    """)

    st.divider()

    if st.session_state.results is None:

        st.subheader("Example Output")

        demo = pd.DataFrame({
            "Risk Probability": [0.08, 0.45, 0.91],
            "Decision": ["AI Automation", "Hybrid", "Human Review"],
            "Expected Cost ($)": [1.2, 3.8, 7.6]
        })

        st.dataframe(demo, use_container_width=True)

    else:

        df = st.session_state.results

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Transactions", f"{len(df):,}")
        col2.metric("Total Cost", f"${df['expected_cost'].sum():,.0f}")
        col3.metric("High Risk", f"{(df['risk_probability'] > 0.7).mean():.1%}")
        col4.metric("Automation", f"{(df['optimal_strategy'].str.contains('AI')).mean():.1%}")

        st.bar_chart(df["optimal_strategy"].value_counts())

# ==============================
# UPLOAD + MAP
# ==============================
elif page == "Upload & Map Data":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Map Columns")

        mapping = {}
        for col in feature_columns + ["order_value"]:
            mapping[col] = st.selectbox(f"{col}", df.columns)

        if st.button("Confirm Mapping"):

            df = df.rename(columns={v: k for k, v in mapping.items()})

            if "helpfulness_ratio" not in df.columns:
                df["helpfulness_ratio"] = 0

            if df["verified_purchase"].dtype == object:
                df["verified_purchase"] = df["verified_purchase"].map({
                    "TRUE": 1, "FALSE": 0, True: 1, False: 0
                })

            st.session_state.mapped_data = df
            st.success("Mapping complete")

# ==============================
# CONFIG
# ==============================
elif page == "Configure":

    st.subheader("Business Settings")

    fraud_cost = st.slider("Fraud Cost Multiplier", 1.0, 5.0, 3.0)
    review_cost = st.slider("Review Cost", 1.0, 20.0, 4.0)

    st.session_state.config = {
        "fraud_cost": fraud_cost,
        "review_cost": review_cost
    }

# ==============================
# RUN ANALYSIS
# ==============================
elif page == "Run Analysis":

    if st.session_state.mapped_data is None:
        st.warning("Upload and map data first")
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
elif page == "Decisions":

    if st.session_state.results is None:
        st.warning("Run analysis first")
    else:
        df = st.session_state.results

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost", f"${df['expected_cost'].sum():,.0f}")
        col2.metric("Automation", f"{(df['optimal_strategy'].str.contains('AI')).mean():.1%}")
        col3.metric("Human Review", f"{(df['optimal_strategy'].str.contains('Human')).mean():.1%}")

        st.divider()

        risk_filter = st.slider("Minimum Risk", 0.0, 1.0, 0.0)

        filtered = df[df["risk_probability"] >= risk_filter]

        cols = ["risk_probability", "risk_tier", "optimal_strategy", "expected_cost"]

        if "top_risk_drivers" in df.columns:
            cols.append("top_risk_drivers")

        if "decision_explanation" in df.columns:
            cols.append("decision_explanation")

        st.dataframe(style_decisions(filtered[cols]), use_container_width=True)

        st.download_button(
            "Download Results",
            filtered.to_csv(index=False),
            "decision_output.csv"
        )

# ==============================
# INSIGHTS
# ==============================
elif page == "Insights":

    if st.session_state.results is None:
        st.warning("Run analysis first")
    else:
        df = st.session_state.results

        st.subheader("Risk Distribution")
        st.bar_chart(df["risk_tier"].value_counts())

        st.subheader("Decision Breakdown")
        st.bar_chart(df["optimal_strategy"].value_counts())

        st.subheader("Cost by Strategy")
        st.dataframe(
            df.groupby("optimal_strategy")["expected_cost"].sum().reset_index(),
            use_container_width=True
        )
