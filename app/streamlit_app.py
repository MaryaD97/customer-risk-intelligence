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

/* ===== BASE ===== */
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

/* ===== TYPOGRAPHY ===== */
h1 {
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: -0.3px;
}
h2 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
h3 {
    font-size: 1.1rem;
    color: #9CA3AF;
}

/* ===== SECTIONS ===== */
.section {
    padding: 1.2rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #1F2937;
}

/* ===== METRICS ===== */
[data-testid="metric-container"] {
    background-color: #0F172A;
    border: 1px solid #1F2937;
    padding: 14px;
    border-radius: 10px;
}

/* ===== BUTTON ===== */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    border: none;
}
.stButton>button:hover {
    background-color: #1D4ED8;
}

/* ===== TABLE ===== */
[data-testid="stDataFrame"] {
    border: 1px solid #1F2937;
    border-radius: 10px;
}

.stDataFrame div[data-testid="stDataFrame"] table {
    font-size: 0.9rem;
}

.stDataFrame div[data-testid="stDataFrame"] table {
    font-size: 0.9rem;
}

/* ===== SUBTEXT ===== */
.caption {
    color: #6B7280;
    font-size: 0.85rem;
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

if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step not in [1,2,3,4,5]:
    st.session_state.step = 1


# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    for col in feature_columns + ["order_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_before = df.isna().mean().mean()

    df = df.fillna(0)
    df["order_value"] = df["order_value"].clip(lower=1)
    df = df.replace([np.inf, -np.inf], 0)

    missing_after = df.isna().mean().mean()

    return df, missing_before, missing_after

def get_risk_drivers(row):
    drivers = []

    if row["order_value"] > 100:
        drivers.append("High transaction value")

    if row["rating"] < 3:
        drivers.append("Low customer rating")

    if row["review_length"] < 20:
        drivers.append("Low engagement activity")

    if row["verified_purchase"] == 0:
        drivers.append("Unverified purchase")

    if not drivers:
        drivers.append("No strong risk signals")

    return drivers[:3]

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

def simulate_decisions(df, fraud_cost, review_cost):
    df = df.copy()

    df["cost_ai"] = df.apply(
        lambda x: cost_ai(x["risk_probability"], x["order_value"], fraud_cost), axis=1
    )
    df["cost_human"] = df.apply(
        lambda x: cost_human(x["risk_probability"], x["order_value"], fraud_cost, review_cost), axis=1
    )
    df["cost_hybrid"] = df.apply(
        lambda x: cost_hybrid(x["risk_probability"], x["order_value"], fraud_cost, review_cost), axis=1
    )

    df["optimal_strategy"] = df.apply(choose_strategy, axis=1)
    df["expected_cost"] = df[["cost_ai", "cost_human", "cost_hybrid"]].min(axis=1)

    return df

def map_action(strategy):
    if strategy == "AI Automation":
        return "Approve"
    elif strategy == "Human Review":
        return "Review"
    else:
        return "Conditional"

def generate_reason(row):
    if row["risk_probability"] > 0.7:
        return "High risk transaction"
    elif row["risk_probability"] > 0.4:
        return "Moderate risk pattern"
    else:
        return "Low risk profile"

def estimate_baseline_cost(df):
    return df["cost_human"].sum()

steps = [
    "Upload Data",
    "Set Costs",
    "Generate Decisions",
    "Decisions",
    "Insights"
]

current_step = st.session_state.step

progress_text = " → ".join([
    f"**{s}**" if i+1 == current_step else s
    for i, s in enumerate(steps)
])

st.caption(f"Step {current_step} of 5")
st.markdown("---")

# ==============================
# OVERVIEW PAGE
# ==============================
if st.session_state.step == 1:

    # ------------------------------
    # PRODUCT MESSAGE (CLARITY)
    # ------------------------------
    st.markdown("## Make better fraud decisions, not just predictions")
    st.markdown("""
    Automatically choose the lowest-cost action for each transaction by balancing fraud risk and operational cost — reviewing high-risk cases while safely approving low-risk ones.
    """)

    # ------------------------------
    # HERO SECTION
    # ------------------------------
    if st.session_state.results is not None:

        df = st.session_state.results

        baseline = estimate_baseline_cost(df)
        optimized = df["expected_cost"].sum()
        savings = baseline - optimized
        reduction = (savings / baseline) if baseline > 0 else 0
        automation = (df["optimal_strategy"].str.contains("AI")).mean()

        st.markdown(f"## 💰 You saved ${savings:,.0f}")
        st.caption("Compared to reviewing all transactions manually")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Cost", f"${baseline:,.0f}")
        col2.metric("Optimized Cost", f"${optimized:,.0f}")
        col3.metric("Loss Reduction", f"{reduction:.1%}")

        col4, col5 = st.columns(2)
        col4.metric("Automation Rate", f"{automation:.1%}")
        st.caption("Baseline = reviewing all transactions manually. Optimized = using AI + selective review to reduce total cost.")

    else:
        st.markdown("## 💰 Start optimizing fraud decisions")

        st.markdown("""
        <div style='padding:16px;border:1px solid #1F2937;border-radius:10px;background:#111827'>
        No data loaded yet.
        </div>
        """, unsafe_allow_html=True)
    
    # ------------------------------
    # SIMPLE FLOW
    # ------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### How decisions are made")

    flow_cols = st.columns(7)

    flow_cols[0].markdown("**Upload Data**")
    flow_cols[1].markdown(
        "<div style='text-align:center;'>→</div>",
        unsafe_allow_html=True
    )
    flow_cols[2].markdown("**Detect Risk**")
    flow_cols[3].markdown(
        "<div style='text-align:center;'>→</div>",
        unsafe_allow_html=True
    )
    flow_cols[4].markdown("**Simulate Cost**")
    flow_cols[5].markdown(
        "<div style='text-align:center;'>→</div>",
        unsafe_allow_html=True
    )
    flow_cols[6].markdown("**Recommend Action**")

    st.divider()

    # ------------------------------
    # OUTPUT PREVIEW
    # ------------------------------

    st.markdown("### Example Decisions")
    st.caption("Recommended action and expected cost per transaction")

    preview_df = pd.DataFrame({
        "Transaction": ["#123", "#124", "#125"],
        "Fraud Risk Score": ["0.89", "0.52", "0.12"],
        "Decision": ["Review", "Conditional", "Approve"],
        "Why": [
            "High value, low trust signal",
            "Moderate risk pattern",
            "Low risk profile"
        ],
        "Expected Cost": ["$12.40", "$6.20", "$1.10"]
    })

    st.dataframe(preview_df, use_container_width=True, height=220)

    st.divider()

    # ------------------------------
    # CTA
    # ------------------------------

    st.subheader("Get Started")

    st.markdown(
        "Upload your dataset and start generating optimized decisions."
    )

    col1, col2, col3 = st.columns([1,1,1])

    col1.markdown("**1. Upload Data**")
    col2.markdown("**2. Set Costs**")
    col3.markdown("**3. Generate Decisions**")

    if st.session_state.results is None:
        st.markdown("""
        <div style='padding:16px;border:1px solid #1F2937;border-radius:10px;background:#111827'>
        Upload your dataset or use sample data to see how decisions are optimized.
        </div>
        """, unsafe_allow_html=True)

    # ==============================
    # UPLOAD
    # ==============================

    st.subheader("What You Need")
    st.caption("We use this data to estimate fraud risk and simulate the cost of each possible decision.")

    st.markdown("""

    **Required:**
    - Customer score or rating
    - Behavioral signal (e.g. review or activity)
    - Transaction value

    The system will:
    - Detect fraud risk
    - Estimate financial impact
    - Recommend the best action
    """)

    st.divider()

    file = st.file_uploader("Upload CSV")

    col1, col2 = st.columns([1, 3])

    with col1:
        sample_clicked = st.button("Try Sample Data")

    with col2:
        st.caption("No setup needed")

    if sample_clicked:

        df = pd.read_csv("sample_data.csv")

        required_cols = feature_columns + ["order_value"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required fields: {', '.join(missing_cols)}")
            st.stop()
        df = df[required_cols]

        df, _, _ = clean_data(df)

        st.session_state.mapped_data = df
        st.session_state.step = 2
        st.rerun()

        st.success(
            "Sample data loaded. You can now generate decisions."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Sample Data Preview")
        st.dataframe(df.head(), use_container_width=True)

    if file:

        df = pd.read_csv(file)
        if df.empty:
            st.error("Uploaded file is empty")
            st.stop()
        
        if len(df.columns) < 2:
            st.error("File does not contain enough usable data")
            st.stop()

        st.subheader("Preview Your Data")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()

        schema_signature = tuple(sorted(df.columns))

        if "saved_mappings" not in st.session_state:
            st.session_state.saved_mappings = {}

        previous_mapping = st.session_state.saved_mappings.get(
            schema_signature, {}
        )

        def suggest_column(target, columns):

            target = target.lower()

            for col in columns:
                if target in col.lower():
                    return col

            return columns[0]

        st.subheader("Match Your Columns")

        st.caption(
            "We’ll automatically match similar datasets in the future."
        )

        st.markdown("Match your data to the required fields")

        mapping = {}

        feature_labels = {
            "rating": "Customer Score",
            "sentiment_score": "Behavioral Signal",
            "review_length": "Engagement Depth",
            "helpfulness_ratio": "Peer Validation",
            "verified_purchase": "Trust Indicator",
            "order_value": "Transaction Value"
        }

        left, right = st.columns(2)

        for i, target_col in enumerate(
            feature_columns + ["order_value"]
        ):

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

        def validate_mapping(mapping, df):

            errors = []

            if len(set(mapping.values())) < len(mapping.values()):
                errors.append(
                    "Duplicate columns selected for multiple fields"
                )

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

        if st.button("Confirm & Continue"):

            if validation_errors:
                st.error("Fix mapping errors before proceeding")

            else:

                df = df.rename(
                    columns={v: k for k, v in mapping.items()}
                )

                required_cols = feature_columns + ["order_value"]
                df = df[required_cols]

                df, before, after = clean_data(df)

                st.session_state.saved_mappings[
                    schema_signature
                ] = mapping

                st.session_state.mapped_data = df
                st.session_state.step = 2
                st.rerun()

                st.success("Data ready for analysis")

                st.subheader("Data Quality Report")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Missing Values Before",
                    f"{before:.2%}"
                )

                col2.metric(
                    "Missing Values After",
                    f"{after:.2%}"
                )

                st.subheader("Cleaned Data Preview")

                st.dataframe(
                    df.head(),
                    use_container_width=True
                )

# END OF STEP 1

# ==============================
# CONFIG
# ==============================
if st.session_state.step == 2:

    st.button("← Back", on_click=lambda: st.session_state.update(step=1))

    if st.session_state.mapped_data is None:
        st.warning("Upload data first to continue")
        st.stop()

    st.title("Define Cost Impact")
    st.caption("Tell the system how expensive fraud and reviews are")

    col1, col2 = st.columns(2)

    fraud_cost = col1.slider(
        "Fraud Loss Impact",
        1.0,
        5.0,
        st.session_state.config["fraud_cost"]
    )
    review_cost = col2.slider(
        "Manual Review Cost",
        1.0,
        20.0,
        st.session_state.config["review_cost"]
    )

    st.session_state.config = {
        "fraud_cost": fraud_cost,
        "review_cost": review_cost
    }


    st.button(
        "Continue to Generate Decisions",
        on_click=lambda: st.session_state.update(step=3)
    )

# ==============================
# RUN
# ==============================
elif st.session_state.step == 3:

    st.button("← Back", on_click=lambda: st.session_state.update(step=2))

    st.title("Generate Decisions")
    st.caption("We’ll analyze risk and recommend lowest-cost actions")

    if st.session_state.mapped_data is None:
        st.warning("Upload your data to continue")
        st.stop()
        
    else:
        st.markdown("<br>", unsafe_allow_html=True)
    
        generate = st.button("Generate Decisions")

        if generate:
        
            with st.spinner("Analyzing transactions..."):
                st.markdown("<div class='caption'>Detecting risk patterns</div>", unsafe_allow_html=True)
        
                df = st.session_state.mapped_data.copy()
                cfg = st.session_state.config
        
                X = df[feature_columns]
                try:
                    df["risk_probability"] = model.predict_proba(X)[:, 1]
                except Exception:
                    st.error("Unable to analyze this dataset. Please check your input format.")
                    st.stop()
        
                st.markdown("<div class='caption'>Optimizing decisions</div>", unsafe_allow_html=True)
        
                df = simulate_decisions(
                    df,
                    cfg["fraud_cost"],
                    cfg["review_cost"]
                )
        
                df["risk_tier"] = df["risk_probability"].apply(risk_tier)
        
                st.session_state.results = df
        
            # 🚀 FORCE UI UPDATE
            st.session_state.step = 4
            st.rerun()
# ==============================
# DECISIONS
# ==============================
elif st.session_state.step == 4:

    st.button("← Back", on_click=lambda: st.session_state.update(step=3))

    if st.session_state.results is None:
        st.warning("Generate decisions first")
        st.stop()

    st.title("Optimized Decisions")

    st.subheader("Adjust Costs")
    
    col1, col2 = st.columns(2)
    
    sim_fraud = col1.slider(
        "Fraud Cost",
        1.0,
        5.0,
        st.session_state.config["fraud_cost"]
    )
    
    sim_review = col2.slider(
        "Review Cost",
        1.0,
        20.0,
        st.session_state.config["review_cost"]
    )
    
    # ✅ FIXED INDENTATION STARTS HERE
    base_df = st.session_state.results
    sim_df = simulate_decisions(base_df, sim_fraud, sim_review)
    df["expected_cost"] = df["expected_cost"].clip(upper=1e6)
    
    st.divider()
    
    total_cost = sim_df["expected_cost"].sum()
    automation_rate = (sim_df["optimal_strategy"].str.contains("AI")).mean()
    high_risk = (sim_df["risk_probability"] > 0.7).mean()
    
    baseline = estimate_baseline_cost(sim_df)
    savings = baseline - total_cost
    
    st.markdown(f"### 💰 Savings: ${savings:,.0f}")

    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Estimated Cost", f"${total_cost:,.0f}")
    col2.metric("Automation Rate", f"{automation_rate:.1%}")
    col3.metric("High Risk Cases", f"{high_risk:.1%}")
    
    st.divider()

    
    
    display_df = sim_df.copy()
    display_df = display_df.reset_index().rename(columns={"index": "Transaction ID"})
    if display_df.empty:
        st.warning("No valid transactions to display")
        st.stop()

    st.markdown("### Recommended Actions")

    sort_option = st.selectbox(
        "Sort by",
        [
            "Default Order",
            "Highest Risk",
            "Highest Cost",
            "Lowest Cost"
        ]
    )

    # Apply sorting BEFORE formatting
    if sort_option == "Highest Risk":
        display_df = display_df.sort_values(by="risk_probability", ascending=False)
    elif sort_option == "Highest Cost":
        display_df = display_df.sort_values(by="expected_cost", ascending=False)
    elif sort_option == "Lowest Cost":
        display_df = display_df.sort_values(by="expected_cost", ascending=True)
    
    display_df["Decision"] = display_df["optimal_strategy"].apply(map_action)
    display_df["Why"] = display_df.apply(generate_reason, axis=1)
    display_df["Why"] = display_df["Why"].str.capitalize()
    
    display_df = display_df[
        [
            "Transaction ID",
            "Decision",
            "expected_cost",
            "risk_probability",
            "Why"
        ]
    ]
    
    display_df.columns = [
        "Transaction ID",
        "Recommended Action",
        "Expected Cost",
        "Risk Score",
        "Why"
    ]
        
    display_df["Risk Score"] = display_df["Risk Score"].map(
        lambda x: f"{x:.2f} ({risk_tier(x)})"
    )
    display_df["Expected Cost"] = display_df["Expected Cost"].map(lambda x: f"${x:,.0f}")

    st.caption("Each action is chosen to minimize cost for that transaction")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=480,
        hide_index=True
    )
    
    
    st.subheader("Decision Rationale")
    
    selected_index = st.selectbox(
        "Select Transaction",
        range(len(sim_df))
    )
    
    row = sim_df.iloc[selected_index]
    
    row = sim_df.loc[selected_index]
    
    st.markdown(f"""
    **Decision Summary**
    - **Action:** {map_action(row['optimal_strategy'])}
    - **Expected Cost:** ${row['expected_cost']:.2f}
    - **Risk Score:** {row['risk_probability']:.2f}
    """)
    
    st.markdown("**Top Risk Drivers**")
    
    drivers = get_risk_drivers(row)
    
    for d in drivers:
        st.markdown(f"- {d}")

    
    st.button(
        "View Insights",
        on_click=lambda: st.session_state.update(step=5)
    )
# ==============================
# INSIGHTS
# ==============================
elif st.session_state.step == 5:

    st.button("← Back", on_click=lambda: st.session_state.update(step=4))

    if st.session_state.results is None:
        st.warning("Generate decisions first")
        st.stop()

    st.title("Business Impact")

    if st.session_state.results is None:
        st.warning("Generate decisions first")
    else:
        df = st.session_state.results

        baseline = estimate_baseline_cost(df)
        optimized = df["expected_cost"].sum()
        savings = baseline - optimized
        reduction = (savings / baseline) if baseline > 0 else 0
        
        # HERO VALUE
        st.markdown(f"## 💰 ${savings:,.0f} saved")
        st.caption("Compared to reviewing all transactions manually")
        
        st.subheader("Business Impact")
        
        col1, col2, col3 = st.columns(3)
        col2.metric("Baseline Cost", f"${baseline:,.0f}")
        col3.metric("Optimized Cost", f"${optimized:,.0f}")
        
        # SECONDARY
        col4, col5 = st.columns(2)
        col4.metric("Loss Reduction", f"{reduction:.1%}")
        col5.metric("Automation Rate", f"{(df['optimal_strategy'].str.contains('AI')).mean():.1%}")

        st.subheader("Key Outcomes")

        st.markdown(f"""
        - Reduced expected loss by **{reduction:.1%}**
        - Automated **{(df['optimal_strategy'].str.contains('AI')).mean():.1%}** of decisions
        - Focused human review on highest-risk transactions
        """)
        
