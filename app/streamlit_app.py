import streamlit as st
import pandas as pd
import joblib
import numpy as np


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

st.title("Customer Risk Intelligence Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Risk Dashboard", "Transaction Explorer", "Live Prediction"]
)

# ==============================
# Risk Dashboard
# ==============================

if page == "Risk Dashboard":

    st.header("Risk Distribution")

    col1, col2, col3 = st.columns(3)

    avg_risk = data["risk_probability"].mean()
    high_risk = (data["risk_probability"] > HYBRID_THRESHOLD).sum()
    total = len(data)

    col1.metric("Average Risk Probability", f"{avg_risk:.3f}")
    col2.metric("High Risk Transactions", high_risk)
    col3.metric("Total Transactions", total)

    st.subheader("Risk Probability Distribution")
    st.bar_chart(data["risk_probability"])

    st.subheader("Strategy Distribution")
    strategy_counts = data["optimal_strategy"].value_counts()
    st.bar_chart(strategy_counts)

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

    filtered = data[data["risk_probability"] >= risk_filter]

    st.dataframe(
        filtered[
            [
                "risk_probability",
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

    verified_purchase = col2.selectbox("Verified Purchase", [0, 1])
    helpfulness_ratio = col2.slider("Helpfulness Ratio", 0.0, 1.0, 0.5)

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([{
            "rating": rating,
            "sentiment_score": sentiment_score,
            "review_length": review_length,
            "verified_purchase": verified_purchase,
            "helpfulness_ratio": helpfulness_ratio
        }])

        processed = pipeline.transform(input_df)
        risk_prob = model.predict_proba(processed)[0][1]

        st.subheader("Prediction Result")
        st.write(f"Risk Probability: **{risk_prob:.3f}**")

        if risk_prob < HYBRID_THRESHOLD:
            strategy = "AI Automation"
        elif risk_prob < 0.7:
            strategy = "Hybrid Handling"
        else:
            strategy = "Human Review"

        st.write(f"Recommended Strategy: **{strategy}**")
