import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Load Artifacts
# ==============================

model = joblib.load("models/risk_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
decision_config = joblib.load("models/decision_config.pkl")

data = pd.read_csv("data/decision_engine_output_subsample.csv")

HYBRID_THRESHOLD = decision_config["HYBRID_THRESHOLD"]

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="Customer Risk Intelligence",
    layout="wide"
)

st.title("Customer Risk Intelligence Platform")

page = st.sidebar.selectbox(
    "Navigation",
    ["Risk Dashboard", "Transaction Explorer", "Live Prediction"]
)

# ==============================
# Risk Tier Classification
# ==============================

def risk_tier(prob):

    if prob < 0.3:
        return "Low Risk"

    elif prob < 0.7:
        return "Medium Risk"

    else:
        return "High Risk"


data["risk_tier"] = data["risk_probability"].apply(risk_tier)

# ==============================
# Risk Dashboard
# ==============================

if page == "Risk Dashboard":

    st.header("System Risk Overview")

    col1, col2, col3, col4 = st.columns(4)

    avg_risk = data["risk_probability"].mean()
    high_risk = (data["risk_tier"] == "High Risk").sum()
    total = len(data)

    automation_rate = (
        data["optimal_strategy"] == "AI Automation"
    ).mean()

    col1.metric("Average Risk", f"{avg_risk:.3f}")
    col2.metric("High Risk Transactions", high_risk)
    col3.metric("Automation Rate", f"{automation_rate:.2%}")
    col4.metric("Total Transactions", total)

    st.divider()

    # ==============================
    # Risk Tier Distribution
    # ==============================

    st.subheader("Risk Tier Distribution")

    tier_counts = data["risk_tier"].value_counts()

    st.bar_chart(tier_counts)

    # ==============================
    # Strategy Distribution
    # ==============================

    st.subheader("Operational Strategy Allocation")

    strategy_counts = data["optimal_strategy"].value_counts()

    st.bar_chart(strategy_counts)

    # ==============================
    # Financial Exposure
    # ==============================

    if "expected_cost" in data.columns:

        st.subheader("Estimated Financial Exposure")

        total_cost = data["expected_cost"].sum()

        st.metric(
            "Estimated System Cost",
            f"${total_cost:,.2f}"
        )

        cost_by_strategy = (
            data.groupby("optimal_strategy")["expected_cost"]
            .sum()
        )

        st.bar_chart(cost_by_strategy)

    # ==============================
    # Risk Heatmap
    # ==============================

    st.subheader("Risk Correlation Heatmap")

    numeric_cols = data.select_dtypes(include=np.number)

    fig, ax = plt.subplots()

    sns.heatmap(
        numeric_cols.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ==============================
# Transaction Explorer
# ==============================

elif page == "Transaction Explorer":

    st.header("Transaction Explorer")

    risk_filter = st.slider(
        "Minimum Risk Probability",
        0.0,
        1.0,
        0.0
    )

    strategy_filter = st.multiselect(
        "Strategy Filter",
        options=data["optimal_strategy"].unique(),
        default=data["optimal_strategy"].unique()
    )

    filtered = data[
        (data["risk_probability"] >= risk_filter)
        &
        (data["optimal_strategy"].isin(strategy_filter))
    ]

    st.dataframe(
        filtered[
            [
                "risk_probability",
                "risk_tier",
                "optimal_strategy",
                "top_risk_drivers",
                "decision_explanation"
            ]
        ],
        use_container_width=True
    )

# ==============================
# Live Prediction
# ==============================

else:

    st.header("Live Risk Prediction")

    col1, col2 = st.columns(2)

    rating = col1.slider("Rating", 1.0, 5.0, 4.0)
    sentiment_score = col1.slider("Sentiment Score", -1.0, 1.0, 0.5)
    review_length = col1.number_input("Review Length", 1, 1000, 100)

    verified_purchase = col2.selectbox(
        "Verified Purchase",
        [0, 1]
    )

    helpfulness_ratio = col2.slider(
        "Helpfulness Ratio",
        0.0,
        1.0,
        0.5
    )

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([{
            "rating": rating,
            "sentiment_score": sentiment_score,
            "review_length": review_length,
            "verified_purchase": verified_purchase,
            "helpfulness_ratio": helpfulness_ratio
        }])

        risk_prob = model.predict_proba(input_df)[0][1]

        tier = risk_tier(risk_prob)

        st.subheader("Prediction Result")

        st.write(f"Fraud Risk Probability: **{risk_prob:.3f}**")
        st.write(f"Risk Tier: **{tier}**")

        if risk_prob < HYBRID_THRESHOLD:
            strategy = "AI Automation"
        elif risk_prob < 0.7:
            strategy = "Hybrid Handling"
        else:
            strategy = "Human Review"

        st.write(f"Recommended Strategy: **{strategy}**")
