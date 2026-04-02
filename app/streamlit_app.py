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
st.sidebar.title("Customer Risk Intelligence Platform")

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

    st.title("Customer Risk Intelligence Platform")

    st.markdown("""
    Automatically choose the **lowest-cost action** for every transaction by combining fraud prediction with financial impact modeling.
    """)

    st.divider()

# ==============================
# EXECUTIVE SUMMARY (NEW)
# ==============================
if st.session_state.results is not None:
    df = st.session_state.results

    baseline = estimate_baseline_cost(df)
    optimized = df["expected_cost"].sum()
    savings = baseline - optimized

    st.subheader("Executive Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Cost", f"${baseline:,.0f}")
    col2.metric("Optimized Cost", f"${optimized:,.0f}")
    col3.metric("Cost Savings", f"${savings:,.0f}")

    st.divider()

# ==============================
# SYSTEM PIPELINE (NEW)
# ==============================
st.subheader("System Pipeline")

st.markdown("""
1. **Data Ingestion**  
   Upload transaction and customer interaction data  

2. **Feature Engineering**  
   Behavioral signals derived from raw inputs  

3. **Risk Modeling**  
   Machine learning model predicts probability of risk  

4. **Decision Optimization**  
   Cost simulation evaluates possible actions  

5. **Explainability Layer**  
   Key drivers behind each decision are surfaced  
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

    # ==============================
    # DATA REQUIREMENTS (NEW)
    # ==============================
    st.subheader("Data Requirements")
    
    st.markdown("""
    Your dataset should include the following types of information:
    
    **Core Signals (Required)**
    - Customer rating or satisfaction score  
    - Review or feedback text  
    - Purchase verification indicator  
    - Engagement signals (e.g., helpful votes)  
    
    **Financial Data (Required)**
    - Transaction value / order amount  
    
    **What the system does:**
    - Derives behavioral features automatically  
    - Predicts risk probability  
    - Recommends cost-optimized decisions  
    """)
    
    st.divider()

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()

        # ==============================
        # SCHEMA SIGNATURE (for learning)
        # ==============================
        schema_signature = tuple(sorted(df.columns))

        if "saved_mappings" not in st.session_state:
            st.session_state.saved_mappings = {}

        previous_mapping = st.session_state.saved_mappings.get(schema_signature, {})

        # ==============================
        # SMART COLUMN SUGGESTION
        # ==============================
        def suggest_column(target, columns):
            target = target.lower()
            for col in columns:
                if target in col.lower():
                    return col
            return columns[0]

        st.subheader("Column Mapping")

        st.caption("The system remembers your schema and will auto-map fields for similar datasets.")

        st.markdown("Map your dataset fields to required features")

        mapping = {}

        # ==============================
        # FRIENDLY FEATURE NAMES (NEW)
        # ==============================
        feature_labels = {
            "rating": "Customer Score",
            "sentiment_score": "Behavioral Signal",
            "review_length": "Engagement Depth",
            "helpfulness_ratio": "Peer Validation",
            "verified_purchase": "Trust Indicator",
            "order_value": "Transaction Value"
        }

        # DRAG-LIKE UX using columns
        left, right = st.columns(2)

        for i, target_col in enumerate(feature_columns + ["order_value"]):

            # Priority: previous mapping → smart suggestion
            default_col = previous_mapping.get(
                target_col,
                suggest_column(target_col, df.columns)
            )

            container = left if i % 2 == 0 else right

            mapping[target_col] = container.selectbox(
                f"{feature_labels.get(target_col, target_col)}",
                df.columns,
                index=list(df.columns).index(default_col)
            )

        st.divider()

        # ==============================
        # VALIDATION SYSTEM
        # ==============================
        def validate_mapping(mapping, df):
            errors = []

            # Duplicate mapping check
            if len(set(mapping.values())) < len(mapping.values()):
                errors.append("Duplicate columns selected for multiple fields")

            # Missing required columns
            for k, v in mapping.items():
                if v not in df.columns:
                    errors.append(f"Missing column: {v}")

            return errors

        validation_errors = validate_mapping(mapping, df)

        if validation_errors:
            st.error("⚠️ Mapping Issues Detected")
            for err in validation_errors:
                st.write(f"- {err}")
        else:
            st.success("✅ Mapping looks good")

        st.divider()

        # ==============================
        # DATA CLEANING PIPELINE
        # ==============================
        def clean_data(df):

            df = df.copy()

            # Standardize column names
            df.columns = df.columns.str.strip()

            # Convert numeric columns
            for col in feature_columns + ["order_value"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Missing values handling
            missing_before = df.isna().mean().mean()

            df = df.fillna(0)

            # Remove infinities
            df = df.replace([np.inf, -np.inf], 0)

            missing_after = df.isna().mean().mean()

            return df, missing_before, missing_after

        # ==============================
        # CONFIRM BUTTON
        # ==============================
        if st.button("Confirm Mapping & Clean Data"):

            if validation_errors:
                st.error("Fix mapping errors before proceeding")
            else:
                # Rename columns
                df = df.rename(columns={v: k for k, v in mapping.items()})

                # Keep only required columns
                required_cols = feature_columns + ["order_value"]
                df = df[required_cols]

                # Clean data
                df, before, after = clean_data(df)

                # SAVE SCHEMA (learning)
                st.session_state.saved_mappings[schema_signature] = mapping

                st.session_state.mapped_data = df

                st.success("Data mapped, validated, and cleaned successfully")

                # ==============================
                # DATA QUALITY INSIGHTS
                # ==============================
                st.subheader("Data Quality Report")

                col1, col2 = st.columns(2)
                col1.metric("Missing Values Before", f"{before:.2%}")
                col2.metric("Missing Values After", f"{after:.2%}")

                st.subheader("Cleaned Data Preview")
                st.dataframe(df.head(), use_container_width=True)

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
