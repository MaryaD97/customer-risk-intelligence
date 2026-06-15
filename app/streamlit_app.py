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

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

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

/* ===== CARD SYSTEM ===== */
.card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
}

/* ===== COLOR SYSTEM ===== */
:root {
    --bg-main: #0B0F17;
    --bg-card: #0F172A;
    --border: #1F2937;
    --text-primary: #E5E7EB;
    --text-secondary: #9CA3AF;
    --green: #22C55E; /* Approve */
    --amber: #F59E0B; /* Review */
    --red: #EF4444; /* High Risk */
    --blue: #3B82F6; /* Cost */
}

/* ===== METRICS ===== */
[data-testid="metric-container"] {
    background-color: #0F172A;
    border: 1px solid #1F2937;
    padding: 14px;
    border-radius: 10px;
}

/* ===== BUTTON ===== */
.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    border: none;
}

.stButton > button:hover {
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
    st.session_state.config = {
        "fraud_cost": 3.0,
        "review_cost": 8.0
    }

if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step not in [1, 2, 3, 4, 5]:
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


REVIEW_EFFECTIVENESS = 0.9  # humans catch ~90% fraud
AI_EFFECTIVENESS = 0.6      # automation catches ~60%


def risk_tier(p):
    return "Low" if p < 0.3 else "Medium" if p < 0.7 else "High"


def cost_ai(p, amt, fraud_cost):
    fraud_loss = (1 - AI_EFFECTIVENESS) * p * amt * fraud_cost
    return fraud_loss


def cost_human(p, amt, fraud_cost, review_cost):
    fraud_loss = (1 - REVIEW_EFFECTIVENESS) * p * amt * fraud_cost
    return review_cost + fraud_loss


def cost_hybrid(p, amt, fraud_cost, review_cost):
    return (
        cost_ai(p, amt, fraud_cost)
        if p < 0.4
        else cost_human(p, amt, fraud_cost, review_cost)
    )


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
        lambda x: cost_ai(
            x["risk_probability"],
            x["order_value"],
            fraud_cost
        ),
        axis=1
    )

    df["cost_human"] = df.apply(
        lambda x: cost_human(
            x["risk_probability"],
            x["order_value"],
            fraud_cost,
            review_cost
        ),
        axis=1
    )

    df["cost_hybrid"] = df.apply(
        lambda x: cost_hybrid(
            x["risk_probability"],
            x["order_value"],
            fraud_cost,
            review_cost
        ),
        axis=1
    )

    df["optimal_strategy"] = df.apply(
        choose_strategy,
        axis=1
    )

    df["expected_cost"] = df[
        ["cost_ai", "cost_human", "cost_hybrid"]
    ].min(axis=1)

    return df


def map_action(strategy):
    if strategy == "AI Automation":
        return "Auto Approve (AI)"
    elif strategy == "Human Review":
        return "Manual Review"
    else:
        return "Unknown"


def generate_reason(row):
    reasons = []

    if row["risk_probability"] > 0.7:
        reasons.append("High fraud risk")

    if row["order_value"] > 100:
        reasons.append("High order value")

    if row["rating"] < 3:
        reasons.append("Low customer trust")

    if row["verified_purchase"] == 0:
        reasons.append("Unverified purchase")

    if row["review_length"] < 20:
        reasons.append("Low activity signal")

    if not reasons:
        return "No strong risk signals"

    return ", ".join(reasons[:2])


def estimate_baseline_cost(df):
    return df["cost_human"].sum()


def format_money(x):
    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    elif x >= 1_000:
        return f"${x/1_000:.2f}K"

    return f"${x:.2f}"

def format_money_short(x):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"${x/1_000:.1f}K"

    return f"${x:.1f}"


steps = [
    "Upload Data",
    "Set Costs",
    "Generate Decisions",
    "Decisions",
    "Insights"
]

current_step = st.session_state.step

progress_text = " → ".join([
    f"**{s}**" if i + 1 == current_step else s
    for i, s in enumerate(steps)
])



st.caption(f"Step {current_step} of 5")
st.markdown("---")

# ==============================
# OVERVIEW PAGE
# ==============================
if st.session_state.step == 1:

    # ==============================
    # STEP 1 — LOAD DATA
    # ==============================
    st.title("Fraud Decision Engine")

    st.markdown("""
    **Decide the lowest-cost action for every transaction.**
    """)

    st.caption(
        "Used to reduce fraud loss while minimizing manual review costs"
    )

    st.markdown("---")

    st.subheader("What You Need")

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

    st.markdown("### Load Data")

    st.caption(
        "Upload your dataset to begin decision analysis"
    )

    # ------------------------------
    # DATA SOURCE
    # ------------------------------
    col1, col2 = st.columns(2)

    with col1:
        file = st.file_uploader("Upload CSV")

    with col2:
        sample_clicked = st.button("Use Sample Data")

    st.caption("Quick start with preloaded dataset")

    # ------------------------------
    # SAMPLE DATA FLOW
    # ------------------------------
    if sample_clicked:

        df = pd.read_csv("sample_data.csv")

        required_cols = feature_columns + ["order_value"]

        missing_cols = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            st.error(
                f"Missing required fields: {', '.join(missing_cols)}"
            )
            st.stop()

        df = df[required_cols]

        df, _, _ = clean_data(df)

        st.session_state.mapped_data = df

        st.session_state.step = 2

        st.success("Sample data loaded successfully")

        st.rerun()

    # ------------------------------
    # FILE UPLOAD FLOW
    # ------------------------------
    if file:

        df = pd.read_csv(file)

        if df.empty:
            st.error("Uploaded file is empty")
            st.stop()

        if len(df.columns) < 2:
            st.error(
                "File does not contain enough usable data"
            )
            st.stop()

        # ------------------------------
        # PREVIEW
        # ------------------------------
        st.markdown("### Data Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.markdown("---")

        # ------------------------------
        # MAPPING
        # ------------------------------
        st.markdown("### Map Required Fields")

        st.caption(
            "Match your dataset columns to required fields below."
        )

        schema_signature = tuple(sorted(df.columns))

        if "saved_mappings" not in st.session_state:
            st.session_state.saved_mappings = {}

        previous_mapping = (
            st.session_state.saved_mappings.get(
                schema_signature,
                {}
            )
        )

        def suggest_column(target, columns):
            target = target.lower()

            for col in columns:
                if target in col.lower():
                    return col

            return columns[0]

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
                suggest_column(
                    target_col,
                    df.columns
                )
            )

            container = left if i % 2 == 0 else right

            mapping[target_col] = container.selectbox(
                feature_labels.get(
                    target_col,
                    target_col
                ),
                df.columns,
                index=list(df.columns).index(default_col)
            )

        # ------------------------------
        # VALIDATION
        # ------------------------------
        def validate_mapping(mapping, df):

            errors = []

            if len(set(mapping.values())) < len(mapping.values()):
                errors.append(
                    "Duplicate columns selected"
                )

            for k, v in mapping.items():
                if v not in df.columns:
                    errors.append(
                        f"Missing column: {v}"
                    )

            return errors

        validation_errors = validate_mapping(
            mapping,
            df
        )

        if validation_errors:

            st.error("Mapping issues detected")

            for err in validation_errors:
                st.write(f"- {err}")

        else:
            st.success("Mapping complete")

        # ------------------------------
        # CONFIRM
        # ------------------------------
        if st.button("Confirm & Continue"):

            if validation_errors:

                st.error(
                    "Fix mapping errors before proceeding"
                )

            else:

                df = df.rename(
                    columns={
                        v: k
                        for k, v in mapping.items()
                    }
                )

                required_cols = (
                    feature_columns + ["order_value"]
                )

                df = df[required_cols]

                df, _, _ = clean_data(df)

                st.session_state.saved_mappings[
                    schema_signature
                ] = mapping

                st.session_state.mapped_data = df

                st.success("Data ready for analysis")

                st.session_state.step = 2

                st.rerun()

# ==============================
# CONFIG
# ==============================
elif st.session_state.step == 2:

    st.button(
        "← Back",
        on_click=lambda: st.session_state.update(step=1)
    )

    if st.session_state.mapped_data is None:
        st.warning("Upload data first to continue")
        st.stop()

    st.title("Set Business Assumptions")

    st.caption(
        "Define the financial impact of fraud and manual review"
    )

    st.caption("""
    Assumptions:
    - Manual review catches ~90% of fraud
    - Automation catches ~60% of fraud

    Automation reduces review cost but allows more fraud loss.
    """)

    col1, col2 = st.columns(2)

    fraud_cost = col1.slider(
        "Fraud Loss Multiplier",
        1.0,
        5.0,
        st.session_state.config["fraud_cost"]
    )

    review_cost = col2.slider(
        "Cost per Manual Review",
        1.0,
        20.0,
        st.session_state.config["review_cost"]
    )

    st.caption("""
    Fraud Loss Multiplier = financial impact when fraud is missed
    (relative to order value)

    Cost per Manual Review = operational cost to investigate one
    transaction
    """)

    st.session_state.config = {
        "fraud_cost": fraud_cost,
        "review_cost": review_cost
    }

    st.button(
        "Run Decision Engine",
        on_click=lambda: st.session_state.update(step=3)
    )

# ==============================
# RUN
# ==============================
elif st.session_state.step == 3:

    st.button(
        "← Back",
        on_click=lambda: st.session_state.update(step=2)
    )

    st.title("Run Decision Engine")

    st.caption(
        "Analyze transactions and compute the lowest-cost action"
    )

    if st.session_state.mapped_data is None:

        st.warning("Upload your data to continue")
        st.stop()

    else:

        st.markdown("<br>", unsafe_allow_html=True)

        generate = st.button("Run Analysis")

        if generate:

            with st.spinner("Running decision engine..."):

                df = st.session_state.mapped_data.copy()

                cfg = st.session_state.config

                X = df[feature_columns]

                try:
                    df["risk_probability"] = (
                        model.predict_proba(X)[:, 1]
                    )

                except Exception:

                    st.error(
                        "Unable to analyze this dataset. "
                        "Please check your input format."
                    )
                    st.stop()

                df = simulate_decisions(
                    df,
                    cfg["fraud_cost"],
                    cfg["review_cost"]
                )

                df["risk_tier"] = df[
                    "risk_probability"
                ].apply(risk_tier)

                st.session_state.results = df

                # 🚀 FORCE UI UPDATE
                st.session_state.step = 4

                st.rerun()

# ==============================
# DECISIONS
# ==============================
elif st.session_state.step == 4:

    st.button(
        "← Back",
        on_click=lambda: st.session_state.update(step=3)
    )

    if st.session_state.results is None:
        st.warning("Generate decisions first")
        st.stop()

    st.title("Recommended Actions")

    st.caption(
        "Each action minimizes expected cost per transaction"
    )

    st.subheader("Adjust Costs")

    col1, col2 = st.columns(2)

    sim_fraud = col1.slider(
        "Fraud Loss Multiplier",
        1.0,
        5.0,
        st.session_state.config["fraud_cost"]
    )

    sim_review = col2.slider(
        "Cost per Manual Review",
        1.0,
        20.0,
        st.session_state.config["review_cost"]
    )

    if st.session_state.mapped_data is not None:

        st.caption(
            f"Data Loaded: Yes | "
            f"Rows: {len(st.session_state.mapped_data)} | "
            f"Fraud Cost: {st.session_state.config['fraud_cost']} | "
            f"Review Cost: {st.session_state.config['review_cost']}"
        )

    # ✅ FIXED INDENTATION STARTS HERE

    base_df = st.session_state.results

    sim_df = simulate_decisions(
        base_df,
        sim_fraud,
        sim_review
    )

    st.session_state.simulated_results = sim_df

    st.divider()

    total_cost = sim_df["expected_cost"].sum()

    automation_rate = (
        sim_df["optimal_strategy"]
        .str.contains("AI")
        .mean()
    )

    baseline = estimate_baseline_cost(sim_df)

    savings = baseline - total_cost

    full_auto_cost = (
        (
            (1 - AI_EFFECTIVENESS)
            * sim_df["risk_probability"]
            * sim_df["order_value"]
            * sim_fraud
        )
    ).sum()

    automation_savings = full_auto_cost - total_cost

    # Card 1
    st.markdown("### Cost Impact")
    
    c1, c2 = st.columns(2)

    c1.metric(
        "Saved vs Manual Review",
        format_money_short(savings)
    )
    
    c2.metric(
        "Saved vs AI-Only",
        format_money_short(automation_savings)
    )
    
    c1, c2 = st.columns(2)
    
    c1.metric(
        "Baseline Cost",
        format_money(baseline)
    )
    
    c2.metric(
        "Optimized Cost",
        format_money(total_cost)
    )

    
    # Card 2

    transactions_analyzed = len(sim_df)
    
    st.markdown("#### Decision Breakdown")
    
    c1, c2 = st.columns(2)
    
    c1.metric(
        "Approved Automatically",
        f"{automation_rate:.1%}"
    )
    
    c2.metric(
        "Requires Review",
        f"{1 - automation_rate:.1%}"
    )

    # ==============================
    # STRATEGY DISTRIBUTION
    # ==============================
    
    st.markdown("#### Strategy Distribution")
    
    strategy_counts = (
        sim_df["optimal_strategy"]
        .value_counts()
    )
    
    dist1, dist2 = st.columns(2)
    
    dist1.metric(
        "AI Automation",
        f"{strategy_counts.get('AI Automation', 0):,}"
    )
    
    dist2.metric(
        "Human Review",
        f"{strategy_counts.get('Human Review', 0):,}"
    )
    
    
    display_df = sim_df.copy()
    
    if "transaction_id" in display_df.columns:
        display_df = display_df.rename(
            columns={"transaction_id": "Transaction ID"}
        )
        id_name = "Transaction ID"
    else:
        display_df = (
            display_df
            .reset_index()
            .rename(columns={"index": "Row ID"})
        )
        id_name = "Row ID"
    
    if display_df.empty:
        st.warning("No valid transactions to display")
        st.stop()
    
    sort_option = st.selectbox(
        "Sort by",
        [
            "Default Order",
            "Highest Risk (Recommended)",
            "Highest Cost",
            "Lowest Cost"
        ]
    )
    
    show_high_risk = st.checkbox(
        "Show high-risk transactions only"
    )
    
    # Apply sorting BEFORE formatting
    
    if sort_option == "Highest Risk (Recommended)":
        display_df = display_df.sort_values(
            by="risk_probability",
            ascending=False
        )
    
    elif sort_option == "Highest Cost":
        display_df = display_df.sort_values(
            by="expected_cost",
            ascending=False
        )
    
    elif sort_option == "Lowest Cost":
        display_df = display_df.sort_values(
            by="expected_cost",
            ascending=True
        )
    
    if show_high_risk:
        display_df = display_df[
            display_df["risk_probability"] >= 0.70
        ]
    
    display_df["Decision"] = (
        display_df["optimal_strategy"]
        .apply(map_action)
    )
    
    display_df["Why"] = (
        display_df
        .apply(generate_reason, axis=1)
    )
    
    display_df["Why"] = (
        display_df["Why"]
        .str.capitalize()
    )
    
    display_df = display_df[
        [
            id_name,
            "Decision",
            "risk_probability",
            "expected_cost",
            "Why"
        ]
    ]
    
    display_df.columns = [
        id_name,
        "Recommended Action",
        "Fraud Risk Score",
        "Expected Cost",
        "Why"
    ]
    
    display_df["Fraud Risk Score"] = (
        display_df["Fraud Risk Score"]
        .map(lambda x: f"{x:.2f} ({risk_tier(x)})")
    )
    
    display_df["Expected Cost"] = (
        display_df["Expected Cost"]
        .map(format_money)
    )

    # ==============================
    # TRANSACTION TABLE
    # ==============================
    
    def color_decision(val):
        if "Auto" in val:
            return "color: #22C55E; font-weight: 600"
        elif "Review" in val:
            return "color: #F59E0B; font-weight: 600"
        return ""
    
    st.markdown("#### Transaction Decisions")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=480,
        hide_index=True
    )
    
    
    # ==============================
    # DECISION EXPLAINER
    # ==============================
    
    st.subheader("Decision Rationale")
    
    selected_index = st.selectbox(
        "Select Transaction",
        range(len(sim_df))
    )
    
    st.markdown("### Decision Breakdown")

    recommended_action = map_action(
        row["optimal_strategy"]
    )
    
    st.markdown(
        f"""
    **Recommended Action:** {recommended_action}
    
    **Expected Cost:** {format_money(row['expected_cost'])}
    
    **Fraud Risk Score:** {row['risk_probability']:.2f}
    """
    )
    
    st.markdown("#### Why This Action Was Chosen")
    
    ai_cost = row["cost_ai"]
    human_cost = row["cost_human"]
    hybrid_cost = row["cost_hybrid"]
    
    if row["optimal_strategy"] == "AI Automation":
    
        explanation = (
            f"AI Automation was selected because its expected cost "
            f"({format_money(ai_cost)}) is lower than Human Review "
            f"({format_money(human_cost)}) and Hybrid "
            f"({format_money(hybrid_cost)})."
        )
    
    elif row["optimal_strategy"] == "Human Review":
    
        explanation = (
            f"Human Review was selected because its expected cost "
            f"({format_money(human_cost)}) is lower than AI Automation "
            f"({format_money(ai_cost)}) and Hybrid "
            f"({format_money(hybrid_cost)})."
        )
    
    else:
    
        explanation = (
            f"Hybrid was selected because its expected cost "
            f"({format_money(hybrid_cost)}) is lower than the other strategies."
        )
    
    st.info(explanation)
    
    st.markdown("#### Key Risk Signals")
    
    drivers = get_risk_drivers(row)
    
    for d in drivers:
        st.markdown(f"- {d}")

# ==============================
# INSIGHTS
# ==============================

elif st.session_state.step == 5:

    df = st.session_state.get(
        "simulated_results",
        st.session_state.results
    )

    st.button(
        "← Back",
        on_click=lambda: st.session_state.update(step=4)
    )

    if st.session_state.results is None:
        st.warning("Generate decisions first")
        st.stop()

    st.title("Value Summary")

    st.caption(
        "Impact based on your current cost settings"
    )

    baseline = estimate_baseline_cost(df)
    optimized = df["expected_cost"].sum()
    
    savings = baseline - optimized
    
    reduction = (
        savings / baseline
        if baseline > 0
        else 0
    )
    
    automation_rate = (
        df["optimal_strategy"]
        .str.contains("AI")
        .mean()
    )
    
    transactions_analyzed = len(df)
    
    transactions_reviewed = (
        df["optimal_strategy"]
        .eq("Human Review")
        .sum()
    )
    
    transactions_automated = (
        transactions_analyzed
        - transactions_reviewed
    )

    # ==============================
    # HERO METRIC
    # ==============================

    st.metric(
        "You Saved",
        format_money(savings)
    )

    st.caption(
        "Comparison against human-review-only and fully automated strategies"
    )

    # ==============================
    # OPERATIONAL IMPACT
    # ==============================
    
    st.subheader("Operations Summary")
    
    transactions_analyzed = len(df)
    
    transactions_reviewed = (
        df["optimal_strategy"]
        .eq("Human Review")
        .sum()
    )
    
    transactions_automated = (
        transactions_analyzed
        - transactions_reviewed
    )
    
    c1, c2, c3 = st.columns(3)
    
    c1.metric(
        "Transactions Analyzed",
        f"{transactions_analyzed:,}"
    )
    
    c2.metric(
        "Automated Decisions",
        f"{transactions_automated:,}"
    )
    
    c3.metric(
        "Manual Reviews",
        f"{transactions_reviewed:,}"
    )

# ==============================
# BUSINESS OUTCOME SUMMARY
# ==============================

st.subheader("Business Outcome Summary")

st.markdown(
    f"""
The decision engine analyzed **{transactions_analyzed:,} transactions** and recommended the lowest-cost action for each case.

**{transactions_automated:,} transactions** were approved automatically, while **{transactions_reviewed:,} transactions** were routed for manual review.

Based on the current business assumptions, expected operating cost decreased from **{format_money(baseline)}** to **{format_money(optimized)}**, generating projected savings of **{format_money(savings)}**.
"""
)

    
    




