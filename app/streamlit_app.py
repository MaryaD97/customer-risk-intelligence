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
# GLOBAL UI FOUNDATION
# ==============================
st.markdown("""
<style>

/* -------------------------
ROOT COLORS
------------------------- */
:root{
--bg-dark:#07141F;
--bg-dark-2:#0B1E2B;
--green:#22C55E;
--green-soft:#DCFCE7;
--white:#FFFFFF;
--border:#E5E7EB;
--text:#0F172A;
--muted:#64748B;
--radius:18px;
}

/* -------------------------
APP BACKGROUND
------------------------- */
[data-testid="stAppViewContainer"]{
background:
linear-gradient(
180deg,
#07141F 0%,
#0A1B29 34%,
#F6F8FA 34%,
#F6F8FA 100%
);
}

/* -------------------------
REMOVE STREAMLIT DEFAULTS
------------------------- */
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

.block-container{
max-width:1450px;
padding-top:0rem;
padding-bottom:3rem;
padding-left:2rem;
padding-right:2rem;
}

/* remove empty top gap */
section.main > div{
padding-top:0rem;
}

/* -------------------------
TYPOGRAPHY
------------------------- */
html, body, [class*="css"]{
font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

h1,h2,h3{
color:white;
margin:0;
}

p, label{
color:#CBD5E1;
}

/* -------------------------
TOP NAVBAR
------------------------- */
.topbar{
background:rgba(7,20,31,.92);
border:1px solid rgba(255,255,255,.06);
padding:16px 24px;
border-radius:18px;
margin-top:12px;
margin-bottom:18px;
display:flex;
justify-content:space-between;
align-items:center;
backdrop-filter: blur(8px);
}

.brand{
font-size:22px;
font-weight:700;
color:white;
}

.steps{
display:flex;
gap:14px;
align-items:center;
}

.step{
display:flex;
align-items:center;
gap:8px;
color:#94A3B8;
font-size:14px;
font-weight:600;
}

.dot{
width:28px;
height:28px;
border-radius:50%;
background:#1E293B;
display:flex;
align-items:center;
justify-content:center;
font-size:13px;
color:white;
}

.active-dot{
background:#22C55E;
color:#04120B;
font-weight:700;
}

/* -------------------------
GENERIC WHITE CARD
------------------------- */
.white-card{
background:white;
border:1px solid #E5E7EB;
border-radius:18px;
padding:24px;
box-shadow:0 10px 30px rgba(0,0,0,.04);
}

/* -------------------------
SECTION SPACING
------------------------- */
.tight-gap{
margin-top:12px;
}

.big-gap{
margin-top:26px;
}

/* -------------------------
BUTTONS
------------------------- */
.stButton > button{
border-radius:12px;
height:46px;
font-weight:600;
border:none;
padding:0 18px;
background:#2563EB;
color:white;
}

.stButton > button:hover{
background:#1D4ED8;
}

/* -------------------------
SLIDERS
------------------------- */
.stSlider{
padding-top:8px;
padding-bottom:8px;
}

/* -------------------------
DATAFRAME
------------------------- */
[data-testid="stDataFrame"]{
border:none !important;
border-radius:16px;
overflow:hidden;
}

/* -------------------------
HERO SECTION
------------------------- */
.hero-shell{
background:radial-gradient(circle at top right,#134E4A 0%, #07141F 55%);
border-radius:22px;
padding:46px;
margin-bottom:28px;
border:1px solid rgba(255,255,255,.05);
min-height:390px;
display:flex;
justify-content:space-between;
gap:30px;
align-items:center;
}

.hero-left{
flex:1.1;
}

.hero-right{
flex:0.9;
display:flex;
justify-content:center;
align-items:center;
}

.hero-title{
font-size:52px;
font-weight:800;
line-height:1.05;
color:white;
margin-bottom:16px;
}

.hero-sub{
font-size:18px;
line-height:1.7;
color:#A7B4C4;
max-width:620px;
margin-bottom:34px;
}

.hero-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:28px;
margin-top:10px;
}

.hero-list-title{
font-size:18px;
font-weight:700;
color:#4ADE80;
margin-bottom:14px;
}

.hero-list{
color:white;
line-height:2;
font-size:15px;
padding-left:18px;
margin:0;
}

/* -------------------------
SHIELD VISUAL
------------------------- */
.shield-wrap{
width:280px;
height:280px;
border-radius:50%;
background:radial-gradient(circle,#0E2F2B,#07141F);
display:flex;
justify-content:center;
align-items:center;
position:relative;
}

.shield{
font-size:110px;
}

/* -------------------------
UPLOAD AREA
------------------------- */
.upload-shell{
margin-top:8px;
}

.upload-title{
font-size:38px;
font-weight:800;
color:#0F172A;
margin-bottom:8px;
}

.upload-sub{
color:#64748B;
font-size:16px;
margin-bottom:26px;
}

.upload-card{
background:white;
border:1px solid #E5E7EB;
border-radius:22px;
padding:28px;
box-shadow:0 10px 24px rgba(0,0,0,.04);
height:100%;
}

.upload-zone{
border:2px dashed #D7DEE7;
border-radius:18px;
padding:42px 20px;
text-align:center;
margin-top:18px;
}

.quick-btn{
margin-top:26px;
}

/* -------------------------
HEADINGS DARK SECTION FIX
------------------------- */
.dark-small{
color:#CBD5E1;
font-size:14px;
font-weight:600;
margin-bottom:8px;
}

/* -------------------------
DASHBOARD KPI CARDS
------------------------- */
.kpi-card{
background:white;
border:1px solid #E5E7EB;
border-radius:22px;
padding:24px;
box-shadow:0 10px 24px rgba(0,0,0,.04);
height:145px;
}

.kpi-label{
font-size:14px;
font-weight:700;
color:#64748B;
margin-bottom:14px;
}

.kpi-value{
font-size:42px;
font-weight:800;
color:#0F172A;
line-height:1;
}

.kpi-green{
color:#16A34A;
}

.kpi-sub{
font-size:13px;
color:#94A3B8;
margin-top:12px;
}

/* -------------------------
ANALYTICS CARD
------------------------- */
.panel-card{
background:white;
border:1px solid #E5E7EB;
border-radius:22px;
padding:24px;
box-shadow:0 10px 24px rgba(0,0,0,.04);
margin-top:18px;
}

/* -------------------------
SECTION TITLE
------------------------- */
.section-title{
font-size:28px;
font-weight:800;
color:#0F172A;
margin-bottom:8px;
}

.section-sub{
color:#64748B;
margin-bottom:18px;
}

/* -------------------------
PANEL CARD (for table + side)
------------------------- */
.panel-card{
background:white;
border-radius:18px;
padding:22px;
border:1px solid #E5E7EB;
box-shadow:0 6px 20px rgba(0,0,0,0.04);
}

/* -------------------------
KPI CARDS
------------------------- */
.kpi-card{
background:white;
border-radius:18px;
padding:20px;
border:1px solid #E5E7EB;
}

.kpi-label{
font-size:13px;
color:#64748B;
margin-bottom:6px;
}

.kpi-value{
font-size:26px;
font-weight:700;
color:#0F172A;
}

.kpi-green{
color:#16A34A;
}

.kpi-sub{
font-size:12px;
color:#94A3B8;
margin-top:4px;
}

/* -------------------------
SECTION HEADERS
------------------------- */
.section-title{
font-size:26px;
font-weight:700;
color:white;
margin-top:10px;
}

.section-sub{
color:#94A3B8;
margin-bottom:18px;
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
# TOP NAVIGATION
# ==============================
def render_topbar(active_step=1):

    labels = {
        1: "Upload Data",
        2: "Set Costs",
        3: "Decisions",
        4: "Insights"
    }

    html = '<div class="topbar">'
    html += '<div class="brand">Fraud Decision Engine</div>'
    html += '<div class="steps">'

    for i in range(1, 5):
        cls = "dot active-dot" if i == active_step else "dot"

        html += (
            f'<div class="step">'
            f'<div class="{cls}">{i}</div>'
            f'<div>{labels[i]}</div>'
            f'</div>'
        )

    html += '</div></div>'

    st.markdown(html, unsafe_allow_html=True)

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

REVIEW_EFFECTIVENESS = 0.9   # humans catch ~90% fraud
AI_EFFECTIVENESS = 0.6       # automation catches ~60%

def risk_tier(p):
    return "Low" if p < 0.3 else "Medium" if p < 0.7 else "High"

def cost_ai(p, amt, fraud_cost):
    fraud_loss = (1 - AI_EFFECTIVENESS) * p * amt * fraud_cost
    return fraud_loss

def cost_human(p, amt, fraud_cost, review_cost):
    fraud_loss = (1 - REVIEW_EFFECTIVENESS) * p * amt * fraud_cost
    return review_cost + fraud_loss

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
        return f"${x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:.0f}"

# ==============================
# OVERVIEW PAGE
# ==============================
if st.session_state.step == 1:
        render_topbar(1)
# ==============================
# STEP 1 — LOAD DATA
# ==============================
        hero_html = """
        <div class="hero-shell">
        
        <div class="hero-left">
        
        <div class="hero-title">Fraud Decision Engine</div>
        
        <div class="hero-sub">
        Decide the lowest-cost action for every transaction — reduce fraud loss while minimizing manual review costs.
        </div>
        
        <div class="hero-grid">
        
        <div>
        <div class="hero-list-title">What You Need</div>
        <ul class="hero-list">
        <li>Customer score or rating</li>
        <li>Behavioral signal</li>
        <li>Transaction value</li>
        </ul>
        </div>
        
        <div>
        <div class="hero-list-title">System Output</div>
        <ul class="hero-list">
        <li>Detect fraud risk</li>
        <li>Estimate financial impact</li>
        <li>Recommend best action</li>
        </ul>
        </div>
        
        </div>
        </div>
        
        <div class="hero-right">
        <div class="shield-wrap">
        <div class="shield">🛡️</div>
        </div>
        </div>
        
        </div>
        """
        
        st.markdown(hero_html, unsafe_allow_html=True)
                
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)

        st.markdown('<div class="upload-title">Upload Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-sub">Upload your dataset to begin decision analysis.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.35, 1])
        
        with col1:
            st.markdown('<div class="upload-card">', unsafe_allow_html=True)
            st.markdown("#### Upload CSV File")
        
            file = st.file_uploader(
                "Upload CSV",
                label_visibility="collapsed"
            )
        
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size:42px;margin-bottom:10px;">☁️</div>
                <div style="font-weight:700;color:#0F172A;">Drag and drop your CSV file here</div>
                <div style="color:#64748B;margin-top:6px;">or click browse</div>
                <div style="color:#94A3B8;margin-top:14px;font-size:13px;">200MB per file</div>
            </div>
            """, unsafe_allow_html=True)
        
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="upload-card">', unsafe_allow_html=True)
            st.markdown("#### Quick Start")
            st.markdown(
                '<div style="color:#64748B;line-height:1.7;">Use our sample dataset to explore the tool immediately.</div>',
                unsafe_allow_html=True
            )
        
            st.markdown('<div class="quick-btn">', unsafe_allow_html=True)
            sample_clicked = st.button("Use Sample Data", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ------------------------------
        # SAMPLE DATA FLOW
        # ------------------------------
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
                st.error("File does not contain enough usable data")
                st.stop()
        
            # ------------------------------
            # PREVIEW
            # ------------------------------
            st.markdown("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
        
            st.markdown("---")
        
            # ------------------------------
            # MAPPING
            # ------------------------------
            st.markdown("### Map Required Fields")
            st.caption("Match your dataset columns to required inputs")
        
            schema_signature = tuple(sorted(df.columns))
        
            if "saved_mappings" not in st.session_state:
                st.session_state.saved_mappings = {}
        
            previous_mapping = st.session_state.saved_mappings.get(schema_signature, {})
        
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
        
            for i, target_col in enumerate(feature_columns + ["order_value"]):
        
                default_col = previous_mapping.get(
                    target_col,
                    suggest_column(target_col, df.columns)
                )
        
                container = left if i % 2 == 0 else right
        
                mapping[target_col] = container.selectbox(
                    feature_labels.get(target_col, target_col),
                    df.columns,
                    index=list(df.columns).index(default_col)
                )
        
            # ------------------------------
            # VALIDATION
            # ------------------------------
            def validate_mapping(mapping, df):
                errors = []
        
                if len(set(mapping.values())) < len(mapping.values()):
                    errors.append("Duplicate columns selected")
        
                for k, v in mapping.items():
                    if v not in df.columns:
                        errors.append(f"Missing column: {v}")
        
                return errors
        
            validation_errors = validate_mapping(mapping, df)
        
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
                    st.error("Fix mapping errors before proceeding")
                else:
                    df = df.rename(columns={v: k for k, v in mapping.items()})
        
                    required_cols = feature_columns + ["order_value"]
                    df = df[required_cols]
        
                    df, _, _ = clean_data(df)
        
                    st.session_state.saved_mappings[schema_signature] = mapping
                    st.session_state.mapped_data = df
        
                    st.success("Data ready for analysis")
        
                    st.session_state.step = 2
                    st.rerun()

# ==============================
# CONFIG
# ==============================
elif st.session_state.step == 2:
        render_topbar(2)

        st.button("← Back", on_click=lambda: st.session_state.update(step=1))
    
        if st.session_state.mapped_data is None:
            st.warning("Upload data first to continue")
            st.stop()
    
        st.title("Set Business Assumptions")
        st.caption("Define the financial impact of fraud and manual review")
    
        st.caption("""
        Assumptions:
        
        - Manual review catches ~90% of fraud  
        - Automation catches ~60% of fraud  
        
        Automation reduces review cost but allows more fraud loss.
        """)
    
        col1, col2 = st.columns(2)
    
        fraud_cost = col1.slider(
            "Fraud Loss Multiplier (impact of missed fraud)",
            1.0, 5.0,
            st.session_state.config["fraud_cost"]
        )
        
        review_cost = col2.slider(
            "Manual Review Cost (cost per transaction review)",
            1.0, 20.0,
            st.session_state.config["review_cost"]
        )
    
    
        st.session_state.config = {
            "fraud_cost": fraud_cost,
            "review_cost": review_cost
        }
    
    
        if st.button("Run Decision Engine →", use_container_width=True):
    
            with st.spinner("Running decision engine..."):
        
                df = st.session_state.mapped_data.copy()
                cfg = st.session_state.config
        
                X = df[feature_columns]
                df["risk_probability"] = model.predict_proba(X)[:, 1]
        
                df = simulate_decisions(
                    df,
                    cfg["fraud_cost"],
                    cfg["review_cost"]
                )
        
                df["risk_tier"] = df["risk_probability"].apply(risk_tier)
        
                st.session_state.results = df
        
            st.session_state.step = 4
            st.rerun()

# ==============================
# DECISIONS
# ==============================
elif st.session_state.step == 4:
        render_topbar(3)

        st.button("← Back", on_click=lambda: st.session_state.update(step=2))
    
        if st.session_state.results is None:
            st.warning("Generate decisions first")
            st.stop()
    
        st.markdown('<div class="section-title">Decision Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Lowest-cost actions for every transaction based on current assumptions.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        sim_fraud = col1.slider(
            "Fraud Loss Multiplier",
            1.0, 5.0,
            st.session_state.config["fraud_cost"]
        )
        
        sim_review = col2.slider(
            "Manual Review Cost",
            1.0, 20.0,
            st.session_state.config["review_cost"]
        )
        
        base_df = st.session_state.results
        sim_df = simulate_decisions(base_df, sim_fraud, sim_review)
        
        total_cost = sim_df["expected_cost"].sum()
        baseline = estimate_baseline_cost(sim_df)
        savings = baseline - total_cost
        automation_rate = (sim_df["optimal_strategy"].str.contains("AI")).mean()
        
        k1, k2, k3 = st.columns(3)
        
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Expected Cost</div>
                <div class="kpi-value">{format_money(total_cost)}</div>
                <div class="kpi-sub">Optimized strategy output</div>
            </div>
            """, unsafe_allow_html=True)
        
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Savings vs Baseline</div>
                <div class="kpi-value kpi-green">{format_money(savings)}</div>
                <div class="kpi-sub">Compared to reviewing all orders</div>
            </div>
            """, unsafe_allow_html=True)
        
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Automation Rate</div>
                <div class="kpi-value">{automation_rate:.1%}</div>
                <div class="kpi-sub">Orders auto-approved</div>
            </div>
            """, unsafe_allow_html=True)
        
        left, right = st.columns([1.4, 1])

        full_auto_cost = (
            (1 - AI_EFFECTIVENESS)
            * sim_df["risk_probability"]
            * sim_df["order_value"]
            * sim_fraud
        ).sum()
        
        with left:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("### Cost Comparison")
        
            st.progress(min(total_cost / max(baseline,1), 1.0))
            st.caption(f"Optimized cost is {total_cost / baseline:.1%} of full review cost")
        
            c1, c2, c3 = st.columns(3)
            c1.metric("Human Review", format_money(baseline))
            c2.metric("AI Only", format_money(full_auto_cost))
            c3.metric("Optimized", format_money(total_cost))
        
            st.markdown('</div>', unsafe_allow_html=True)
        
        with right:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("### Decision Breakdown")
        
            auto_rate = automation_rate
            review_rate = 1 - automation_rate
        
            st.progress(auto_rate)
            st.caption(f"Auto Approved: {auto_rate:.1%}")
        
            st.progress(review_rate)
            st.caption(f"Manual Review: {review_rate:.1%}")
        
            st.markdown('</div>', unsafe_allow_html=True)

        
        
        # -----------------------------
        # CREATE CONSISTENT ID COLUMN
        # -----------------------------
        if "transaction_id" in sim_df.columns:
            sim_df = sim_df.rename(columns={"transaction_id": "Transaction ID"})
            id_name = "Transaction ID"
        else:
            sim_df = sim_df.reset_index().rename(columns={"index": "Row ID"})
            id_name = "Row ID"
        
        # Create display_df AFTER fixing sim_df
        display_df = sim_df.copy()

        # -----------------------------
        # PREP TABLE DATA
        # ----------------------------- 
        decision_counts = (
            display_df["optimal_strategy"]
            .apply(map_action)
            .value_counts(normalize=True)
        )
        
        approve_rate = decision_counts.get("Auto Approve (AI)", 0)
        review_rate = decision_counts.get("Manual Review", 0)
                
        if display_df.empty:
            st.warning("No valid transactions to display")
            st.stop()
    
        sort_option = st.selectbox(
            "Sort by",
            [
                "Original Order",
                "Highest Risk (Recommended)",
                "Highest Cost",
                "Lowest Cost"
            ],
            index=1
        )
    
        # Apply sorting BEFORE formatting
        if sort_option == "Highest Risk (Recommended)":
            display_df = display_df.sort_values(by="risk_probability", ascending=False)
        elif sort_option == "Highest Cost":
            display_df = display_df.sort_values(by="expected_cost", ascending=False)
        elif sort_option == "Lowest Cost":
            display_df = display_df.sort_values(by="expected_cost", ascending=True)
        
        display_df["Decision"] = display_df["optimal_strategy"].apply(map_action)
    
        display_df["Decision"] = display_df["Decision"].replace({
            "Auto Approve (AI)": "✓ Approve",
            "Manual Review": "Review"
        })
        display_df["Why"] = display_df.apply(generate_reason, axis=1)
        display_df["Why"] = display_df["Why"].str.capitalize()
        display_df["Why"] = display_df["Why"].str.replace(",", " •")
    
        display_df["Risk Level"] = display_df["risk_probability"].apply(risk_tier)
        
        display_df = display_df[
            [
                id_name,
                "Decision",
                "Risk Level",
                "risk_probability",
                "expected_cost",
                "Why"
            ]
        ]
        
        display_df.columns = [
            id_name,
            "Decision",
            "Risk Level",
            "Risk Score",
            "Expected Cost",
            "Why"
        ]
        
        display_df["Risk Score"] = display_df["Risk Score"].map(lambda x: f"{x:.2f}")
    
        display_df["Risk Level"] = display_df["Risk Level"].str.upper()
        
        display_df["Expected Cost"] = display_df["Expected Cost"].map(format_money)
        
        # ✅ APPLY STYLING LAST (after column rename)
        
        
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Transaction Decisions")
        
    
        # ==============================
        # TABLE + ANALYST PANEL LAYOUT
        # ==============================
        
        left, right = st.columns([1.6, 1])
        
        # -----------------------------
        # PREP DISPLAY DATA (KEEP YOUR EXISTING LOGIC ABOVE THIS)
        # -----------------------------
        
        # Risk Badge Styling
        def risk_badge(val):
            if val == "HIGH":
                return "🔴 HIGH"
            elif val == "MEDIUM":
                return "🟡 MEDIUM"
            else:
                return "🟢 LOW"
        
        display_df["Risk Level"] = display_df["Risk Level"].apply(risk_badge)
        
        # Decision styling
        display_df["Decision"] = display_df["Decision"].replace({
            "✓ Approve": "✅ Approve",
            "Review": "🛑 Review"
        })
        
        # -----------------------------
        # LEFT: TABLE
        # -----------------------------
        with left:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("### Transactions")
        
            st.dataframe(
                display_df,
                use_container_width=True,
                height=520,
                hide_index=True
            )
        
            st.markdown('</div>', unsafe_allow_html=True)
        
        # -----------------------------
        # RIGHT: ANALYST PANEL
        # -----------------------------
        with right:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("### Analyst View")
        
            options = display_df[id_name].tolist()
        
            selected_id = st.selectbox(
                "Select Transaction",
                options
            )
        
            # FIXED SELECTION (IMPORTANT)
            row = sim_df[sim_df[id_name] == selected_id].iloc[0]
        
            st.markdown("---")
        
            st.markdown(f"""
            **Decision:** {map_action(row['optimal_strategy'])}  
            **Risk Score:** {row['risk_probability']:.2f}  
            **Expected Cost:** {format_money(row['expected_cost'])}
            """)
        
            st.markdown("#### Cost Breakdown")
        
            st.markdown(f"""
            • AI Cost: {format_money(row['cost_ai'])}  
            • Human Cost: {format_money(row['cost_human'])}  
            • Hybrid Cost: {format_money(row['cost_hybrid'])}
            """)
        
            st.markdown("#### Risk Drivers")
        
            drivers = get_risk_drivers(row)
            st.markdown(" • ".join(drivers))
        
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.button(
            "View Insights →",
            use_container_width=True,
            on_click=lambda: st.session_state.update(step=5)
        )
# ==============================
# INSIGHTS
# ==============================
elif st.session_state.step == 5:
    render_topbar(4)



    df = st.session_state.get("simulated_results", st.session_state.results)

    st.button("← Back", on_click=lambda: st.session_state.update(step=4))

    if st.session_state.results is None:
        st.warning("Generate decisions first")
        st.stop()

    st.title("Value Summary")
    st.caption("Impact based on your current cost settings")

    baseline = estimate_baseline_cost(df)
    optimized = df["expected_cost"].sum()
    savings = baseline - optimized
    reduction = (savings / baseline) if baseline > 0 else 0
        
    # HERO VALUE
    st.markdown(f"## 💰 {format_money(savings)} saved")
    st.caption("Using optimized decisioning instead of reviewing all transactions manually")
    
    st.markdown(f"""
    Reduced total cost from **{format_money(baseline)}** → **{format_money(optimized)}**
    """)
        
    st.subheader("Business Impact")
        
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Human Review Strategy", format_money(baseline))
    c2.metric("Optimized Cost", format_money(optimized))
    c3.metric("Loss Reduction", f"{reduction:.1%}")
    c4.metric("Automation Rate", f"{(df['optimal_strategy'].str.contains('AI')).mean():.1%}")

    st.subheader("Key Outcomes")

    automation_rate = (df['optimal_strategy'].str.contains('AI')).mean()

    st.markdown(f"""
    - {format_money(savings)} saved through optimized decisioning  
    - {automation_rate:.1%} of transactions automated  
    - Manual review focused on highest-risk cases  
    """)
