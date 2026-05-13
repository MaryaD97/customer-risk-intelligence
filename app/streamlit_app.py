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
#0A1B29 22%,
#F6F8FA 22%,
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
position: relative;
left: calc(-1 * (100vw - 100%) / 2);
width: 100vw;

background:#07141F;
border-bottom:1px solid rgba(255,255,255,.08);

padding:20px 60px;
margin:0 0 30px 0;

display:flex;
justify-content:space-between;
align-items:center;
}

.brand{
font-size:22px;
font-weight:700;
color:white;
}

.brand-wrap{
display:flex;
align-items:center;
gap:12px;
}

.brand img{
height:28px;
}

.steps{
display:flex;
gap:22px;
align-items:center;
}

.step{
display:flex;
align-items:center;
gap:8px;
color:#94A3B8;
font-size:17px;
font-weight:600;
line-height:1;
}

.dot{
width:30px;
height:30px;
border-radius:50%;
background:#1E293B;
display:flex;
align-items:center;
justify-content:center;
font-size:14px;
color:white;
}

.active-dot{
background:#22C55E;
color:#04120B;
font-weight:700;
box-shadow:0 0 0 3px rgba(34,197,94,0.2);
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
min-height:540px;
border-radius:14px;
overflow:hidden;
border:1px solid #E5E7EB;
}

/* -------------------------
HERO SECTION
------------------------- */
.hero-shell{
position: relative;

background:
linear-gradient(90deg, #07141F 0%, #0B1E2B 100%);

padding:70px 80px;
margin-bottom:40px;

display:flex;
justify-content:space-between;
align-items:center;

overflow:visible;
min-height:auto;
}

.hero-shell.small{
padding:32px 60px;
min-height:auto;
margin-bottom:30px;

display:flex;
align-items:center;
}

.hero-shell.small .hero-title{
font-size:36px;
}

.hero-shell.small .hero-sub{
font-size:16px;
margin-bottom:0;
max-width:600px;
}

.hero-assumptions{
margin-top:18px;
color:#94A3B8;
font-size:15px;
line-height:1.8;
max-width:700px;
}

.hero-top{
display:flex;
justify-content:space-between;
align-items:center;
gap:30px;
}

.hero-image{
position:absolute;
right:60px;
top:55%;
transform:translateY(-50%);
height:300px;
opacity:0.9;   /* or 1 for full */
}

.hero-left{
flex:1.2;
z-index:2;
}

.hero-right{
flex:0.9;
display:flex;
justify-content:center;
align-items:center;
}

.hero-title{
font-size:64px;
font-weight:800;
line-height:1.05;
letter-spacing:-1px;
}

.hero-sub{
font-size:20px;
line-height:1.8;
color:#A7B4C4;
max-width:650px;
margin-bottom:42px;
}

.hero-grid{
display:flex;
align-items:flex-start;
gap:40px;
margin-top:20px;
}

.hero-list-title{
font-size:18px;
font-weight:700;
color:#4ADE80;
margin-bottom:14px;
}

.hero-list{
color:white;
line-height:2.2;
font-size:16px;
margin:0;
}

.hero-divider{
width:1px;
background:rgba(148,163,184,0.25);
height:140px;
margin:0 10px;
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
border:1.5px dashed #CBD5E1;
border-radius:12px;
padding:30px 20px;
text-align:center;
margin-top:18px;
background:white;
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
FLOATING KPI SYSTEM
------------------------- */

.kpi-row{
position:relative;
margin-top:20px;
margin-bottom:26px;
z-index:15;
}

.kpi-shell{
display:flex;
flex-direction:column;
justify-content:space-between;

background:white;

border:1px solid #E7ECF2;
border-radius:22px;

padding:22px 24px;

height:132px;

box-shadow:
0 10px 30px rgba(15,23,42,.06),
0 2px 8px rgba(15,23,42,.04);

transition:all .18s ease;
}

.kpi-shell:hover{

box-shadow:
0 16px 40px rgba(15,23,42,.10),
0 4px 12px rgba(15,23,42,.06);
}

.kpi-label{
font-size:13px;
font-weight:700;
letter-spacing:.2px;
color:#64748B;
line-height:1.2;
}

.kpi-value{
font-size:38px;
font-weight:800;
letter-spacing:-1px;
line-height:1;

color:#0F172A;

margin-top:10px;
}

.kpi-green{
color:#16A34A;
}

.kpi-sub{
font-size:13px;
font-weight:500;
color:#94A3B8;
line-height:1.3;

margin-top:12px;
}

/* -------------------------
ANALYTICS CARD
------------------------- */
/* -------------------------
ANALYTICS CARD
------------------------- */
.panel-card{
background:white;
border-radius:20px;
padding:20px;
border:1px solid #E5E7EB;
box-shadow:0 6px 20px rgba(0,0,0,0.04);
margin-top:0px;
}

/* -------------------------
DASHBOARD GRID SPACING
------------------------- */
.dashboard-grid{
margin-top:22px;
}

.grid-stack{
display:flex;
flex-direction:column;
gap:20px;
height:100%;
}

[data-testid="column"]{
overflow:visible !important;
}

/* -------------------------
TABLE HEIGHT CONTROL
------------------------- */

.cost-grid{
display:flex;
gap:12px;
margin-top:18px;
}

.mini-card{
flex:1;
background:#F8FAFC;
border:1px solid #E5E7EB;
border-radius:14px;
padding:14px 16px;
}

.mini-label{
font-size:12px;
color:#64748B;
font-weight:600;
margin-bottom:6px;
}

.mini-value{
font-size:20px;
font-weight:800;
color:#0F172A;
}

.mini-card.highlight{
background:#ECFDF5;
border:1px solid #22C55E;
}

.mini-card.highlight .mini-value{
color:#16A34A;
}

/* -------------------------
SECTION TITLE
------------------------- */
.section-title{
font-size:28px;
font-weight:800;
color:#0F172A;
margin-bottom:6px;
}

.section-sub{
color:#64748B;
margin-bottom:20px;
font-size:15px;
}


/* -------------------------
BADGES
------------------------- */
.badge{
padding:4px 10px;
border-radius:999px;
font-size:12px;
font-weight:600;
}

.badge.high{
background:#FEE2E2;
color:#991B1B;
}

.badge.medium{
background:#FEF3C7;
color:#92400E;
}

.badge.low{
background:#DCFCE7;
color:#166534;
}

/* -------------------------
OPERATIONS TABLE
------------------------- */

.ops-table-wrap{
border:1px solid #E5E7EB;
border-radius:18px;
overflow:hidden;
background:white;
margin-top:10px;
}

.ops-table-scroll{
max-height:540px;
overflow-y:auto;
overflow-x:auto;
width:100%;
}

.ops-table{
width:100%;
min-width:900px;
border-collapse:separate;
border-spacing:0;
table-layout:auto;
}

.ops-table thead th{
position:sticky;
top:0;
z-index:5;

background:#F8FAFC;

padding:14px 18px;

font-size:12px;
font-weight:700;
letter-spacing:.3px;
text-transform:uppercase;

color:#64748B;

border-bottom:1px solid #E2E8F0;
}

.ops-table tbody tr{
transition:all .15s ease;
}

.ops-table tbody tr:hover{
background:#F8FAFC;
}

.ops-table tbody td{
padding:16px 18px;

font-size:14px;
font-weight:500;

color:#0F172A;

border-bottom:1px solid #F1F5F9;

vertical-align:middle;
}

/* -------------------------
TABLE TYPOGRAPHY
------------------------- */

.tx-id{
font-weight:700;
color:#0F172A;
}

.reason-cell{
color:#64748B;
font-size:13px;
line-height:1.45;
max-width:320px;
}

/* -------------------------
RISK BADGES
------------------------- */

.risk-pill{
display:inline-flex;
align-items:center;
justify-content:center;

padding:6px 12px;

border-radius:999px;

font-size:12px;
font-weight:700;

letter-spacing:.3px;
}

.risk-high{
background:#FEE2E2;
color:#B91C1C;
}

.risk-medium{
background:#FEF3C7;
color:#B45309;
}

.risk-low{
background:#DCFCE7;
color:#15803D;
}

/* -------------------------
DECISION BADGES
------------------------- */

.decision-pill{
display:inline-flex;
align-items:center;
justify-content:center;

padding:6px 12px;

border-radius:999px;

font-size:12px;
font-weight:700;

letter-spacing:.2px;
}

.decision-approve{
background:#DCFCE7;
color:#166534;
}

.decision-review{
background:#FEF3C7;
color:#92400E;
}

.decision-decline{
background:#FEE2E2;
color:#991B1B;
}

/* -------------------------
SPACING UTILITIES
------------------------- */
.tight-gap{ margin-top:10px; }
.medium-gap{ margin-top:18px; }
.large-gap{ margin-top:28px; }

/* -------------------------
DATAFRAME POLISH
------------------------- */
[data-testid="stDataFrame"] {
border-radius: 14px;
overflow: hidden;
border: 1px solid #E5E7EB;
}

/* -------------------------
HOVER EFFECTS
------------------------- */
.white-card:hover,
.panel-card:hover{
transition: all .2s ease;
}

.panel-card:hover{
box-shadow:0 10px 24px rgba(0,0,0,.08);
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

    html = """<div class="topbar">

<div class="brand">
Fraud Decision Engine
</div>

<div class="steps">
"""

    for i in range(1, 5):

        cls = "dot active-dot" if i == active_step else "dot"

        html += f"""<div class="step">
<div class="{cls}">{i}</div>
<div>{labels[i]}</div>
</div>
"""

    html += """</div>

</div>
"""

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

if st.session_state.step not in [1,2,3,4]:
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

    p = df["risk_probability"]
    amt = df["order_value"]

    df["cost_ai"] = (1 - AI_EFFECTIVENESS) * p * amt * fraud_cost
    df["cost_human"] = review_cost + (1 - REVIEW_EFFECTIVENESS) * p * amt * fraud_cost

    df["cost_hybrid"] = np.where(
        p < 0.4,
        df["cost_ai"],
        df["cost_human"]
    )

    df["expected_cost"] = df[["cost_ai", "cost_human", "cost_hybrid"]].min(axis=1)

    df["optimal_strategy"] = np.select(
        [
            df["cost_ai"] <= df["cost_human"],
            df["cost_human"] <= df["cost_ai"]
        ],
        [
            "AI Automation",
            "Human Review"
        ],
        default="Hybrid"
    )

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
def render_upload_page():
    render_topbar(1)
    
    hero_html = """<div class="hero-shell">
<div class="hero-left">

<div class="hero-top">
<div>
<div class="hero-title">Fraud Decision Engine</div>
<div class="hero-sub">
Decide the lowest-cost action for every transaction — reduce fraud loss while minimizing manual review costs.
</div>
</div>

<img class="hero-image" src="https://raw.githubusercontent.com/MaryaD97/customer-risk-intelligence/main/shield_01.jpg"/>
</div>

<div class="hero-grid">

<div>
<div class="hero-list-title">What You Need</div>
<div class="hero-list">
👤 Customer score or rating<br>
📊 Behavioral signal<br>
💳 Transaction value
</div>
</div>

<div class="hero-divider"></div>

<div>
<div class="hero-list-title">System Output</div>
<div class="hero-list">
🛡️ Detect fraud risk<br>
📉 Estimate financial impact<br>
✅ Recommend best action
</div>
</div>

</div>

</div>
</div>"""
    
    st.markdown(hero_html, unsafe_allow_html=True)    
            
    # ==============================
    # UPLOAD SECTION (CLEAN LAYOUT)
    # ==============================
    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    
    # TOP ROW (titles aligned horizontally)
    colA, colB = st.columns([1,1])
    
    with colA:
        st.markdown('<div class="upload-title">Upload Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-sub">Upload your dataset to begin decision analysis.</div>', unsafe_allow_html=True)
    
    with colB:
        st.markdown('<div class="upload-title">Quick Start</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-sub">Use sample data instantly.</div>', unsafe_allow_html=True)
        
    
    
    # SECOND ROW (upload + button SAME LEVEL)
    col1, col2 = st.columns([1,1])
    
    with col1:
        file = st.file_uploader(
            "Upload CSV",
            label_visibility="collapsed"
        )
    
    with col2:
        sample_clicked = st.button("Use Sample Data", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECURITY BOX (FIXED)
    st.markdown(
        '<div style="margin-top:10px;background:#F1F5F9;border:1px solid #E2E8F0;padding:10px 14px;border-radius:10px;color:#475569;font-size:13px;width:60%;">'
        '🛡️ Your data is secure and used only for analysis. No data is stored permanently.'
        '</div>',
        unsafe_allow_html=True
    )
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
    
        st.markdown('<div class="medium-gap"></div>', unsafe_allow_html=True)
    
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
def render_costs_page():
    render_topbar(2)

    if st.button("← Back"):
        st.session_state.step = 1
        st.rerun()

    if st.session_state.mapped_data is None:
        st.warning("Upload data first to continue")
        st.stop()

    hero_html = """<div class="hero-shell small">
<div class="hero-left">

<div class="hero-title">Set Decision Costs</div>

<div class="hero-sub">
Define how costly fraud and manual review are — the system will optimize decisions accordingly.
</div>

<div class="hero-assumptions">
• Manual review catches ~90% of fraud <br>
• Automation catches ~60% of fraud <br>
• Automation reduces cost but increases risk exposure
</div>

</div>
</div>
"""
    
    st.markdown(hero_html, unsafe_allow_html=True)

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
    # --- KPI PREVIEW ---
    df_preview = st.session_state.mapped_data.copy()
    cfg = st.session_state.config
    
    # quick simulation
    X = df_preview[feature_columns]
    df_preview["risk_probability"] = model.predict_proba(X)[:, 1]
    
    df_preview = simulate_decisions(
        df_preview,
        fraud_cost,
        review_cost
    )
    
    total_cost = df_preview["expected_cost"].sum()
    baseline = estimate_baseline_cost(df_preview)
    savings = baseline - total_cost
    
    k1, k2, k3 = st.columns(3)
    
    with k1:
        st.markdown(f"""
        <div class="kpi-shell">
            <div class="kpi-label">Projected Cost</div>
            <div class="kpi-value">{format_money(total_cost)}</div>
            <div class="kpi-sub">Based on current inputs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with k2:
        st.markdown(f"""
        <div class="kpi-shell">
            <div class="kpi-label">Potential Savings</div>
            <div class="kpi-value kpi-green">{format_money(savings)}</div>
            <div class="kpi-sub">vs full manual review</div>
        </div>
        """, unsafe_allow_html=True)
    
    with k3:
        st.markdown(f"""
        <div class="kpi-shell">
            <div class="kpi-label">Baseline Cost</div>
            <div class="kpi-value">{format_money(baseline)}</div>
            <div class="kpi-sub">Manual-only strategy</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Run Decision Engine →", use_container_width=True):

        with st.spinner("Running decision engine..."):
    
            df = st.session_state.mapped_data.copy()
            cfg = st.session_state.config
    
            X = df[feature_columns]
            df["risk_probability"] = model.predict_proba(X)[:, 1]
            if "risk_probability" not in df.columns:
                st.error("Risk model failed to generate predictions")
                st.stop()
    
            df = simulate_decisions(
                df,
                cfg["fraud_cost"],
                cfg["review_cost"]
            )
    
            df["risk_tier"] = df["risk_probability"].apply(risk_tier)
    
            st.session_state.results = df
    
        st.session_state.step = 3
        st.rerun()

# ==============================
# DECISIONS
# ==============================
def render_decision_page():
    render_topbar(3)

    # -----------------------------
    # HEADER (DARK HERO)
    # -----------------------------
    header_html = """<div class="hero-shell small">
<div class="hero-left">
<div class="hero-title">Decision Breakdown</div>
<div class="hero-sub">
Optimized actions based on fraud risk and cost assumptions.
</div>
</div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)

    if st.button("← Back"):
        st.session_state.step = 2
        st.rerun()

    if st.session_state.results is None:
        st.warning("Generate decisions first")
        st.stop()

    # -----------------------------
    # BASE DATA (NO SLIDERS HERE)
    # -----------------------------
    sim_df = st.session_state.results.copy()

    total_cost = sim_df["expected_cost"].sum()
    baseline = estimate_baseline_cost(sim_df)
    savings = baseline - total_cost
    reduction = (savings / baseline) if baseline > 0 else 0
    automation_rate = (sim_df["optimal_strategy"].str.contains("AI")).mean()

    # -----------------------------
    # FLOATING KPI ROW
    # -----------------------------
    k1, k2, k3 = st.columns(3, gap="large")

    # --------------------------------
    # KPI 1
    # --------------------------------
    with k1:

        st.markdown(f"""<div class="kpi-shell">

<div class="kpi-label">
Total Cost (Optimized)
</div>

<div class="kpi-value">
{format_money(total_cost)}
</div>

<div class="kpi-sub">
Lowest expected cost
</div>

</div>
""", unsafe_allow_html=True)

    # --------------------------------
    # KPI 2
    # --------------------------------
    with k2:

        st.markdown(f"""<div class="kpi-shell">

<div class="kpi-label">
    Savings vs Human Review
</div>

<div class="kpi-value kpi-green">
    {format_money(savings)}
</div>

<div class="kpi-sub">
    {reduction:.1%} cost reduction
</div>

</div>
""", unsafe_allow_html=True)

    # --------------------------------
    # KPI 3
    # --------------------------------
    with k3:

        st.markdown(f"""<div class="kpi-shell">

<div class="kpi-label">
    Automation Rate
</div>

<div class="kpi-value">
    {automation_rate:.1%}
</div>

<div class="kpi-sub">
    Auto approved transactions
</div>

</div>
""", unsafe_allow_html=True)

    # -----------------------------
    # MAIN DASHBOARD GRID
    # -----------------------------
    top_left, top_right = st.columns([1.9, 1], gap="large")

    # =============================
    # LEFT COLUMN
    # =============================
    with top_left:

        # --------------------------------
        # COST COMPARISON
        # --------------------------------
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)

        st.markdown("### Cost Comparison")

        full_auto_cost = (
            (1 - AI_EFFECTIVENESS)
            * sim_df["risk_probability"]
            * sim_df["order_value"]
            * st.session_state.config["fraud_cost"]
        ).sum()

        st.progress(min(total_cost / max(baseline, 1), 1.0))

        ratio = total_cost / baseline if baseline > 0 else 0

        st.caption(
            f"Optimized cost is {ratio:.1%} of full review cost"
        )

        st.markdown(f"""<div class="cost-grid">

<div class="mini-card">
<div class="mini-label">Human Review</div>
<div class="mini-value">{format_money(baseline)}</div>
</div>

<div class="mini-card">
<div class="mini-label">AI Only</div>
<div class="mini-value">{format_money(full_auto_cost)}</div>
</div>

<div class="mini-card highlight">
<div class="mini-label">Optimized</div>
<div class="mini-value">{format_money(total_cost)}</div>
</div>

</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


    # =============================
    # RIGHT COLUMN
    # =============================
    with top_right:

        # --------------------------------
        # DONUT CHART
        # --------------------------------
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)

        st.markdown("### Decision Split")

        import plotly.express as px

        auto_rate = automation_rate
        review_rate = 1 - automation_rate

        fig = px.pie(
            values=[auto_rate, review_rate],
            names=["Approve (AI)", "Review"],
            hole=0.72
        )

        fig.update_traces(
            textinfo='percent',
            hoverinfo='label+percent',
            marker=dict(colors=["#22C55E", "#FACC15"])
        )

        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------
        # ANALYST VIEW
        # --------------------------------
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        
        st.markdown("### Analyst View")
        
        # PREP DISPLAY DATA FIRST
        display_df = sim_df.copy().reset_index(drop=True)
        
        display_df["Transaction ID"] = display_df.index + 1
        
        display_df["Decision"] = display_df["optimal_strategy"].apply(map_action)
        
        display_df["Risk Level"] = (
            display_df["risk_probability"]
            .apply(risk_tier)
            .str.upper()
        )
        
        display_df["Risk Score"] = (
            display_df["risk_probability"]
            .map(lambda x: f"{x:.2f}")
        )
        
        display_df["Expected Cost"] = (
            display_df["expected_cost"]
            .map(format_money)
        )
        
        display_df["Why"] = (
            display_df.apply(generate_reason, axis=1)
        )
        
        selected_id = st.selectbox(
            "Select Transaction",
            display_df["Transaction ID"]
        )
        
        row = display_df.loc[
            display_df["Transaction ID"] == selected_id
        ].iloc[0]
        
        st.markdown(f"""
        **Decision:** {map_action(row['optimal_strategy'])}  
        
        **Risk Score:** {row['risk_probability']:.2f}  
        
        **Expected Cost:** {format_money(row['expected_cost'])}
        """)
        
        st.markdown("##### Cost Breakdown")
        
        st.markdown(f"""
        AI: {format_money(row['cost_ai'])}  
        
        Human: {format_money(row['cost_human'])}  
        
        Hybrid: {format_money(row['cost_hybrid'])}
        """)
        
        st.markdown("##### Risk Drivers")
        
        st.markdown(
            " • ".join(get_risk_drivers(row))
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
   
    # --------------------------------
    # TRANSACTIONS TABLE
    # --------------------------------
    
    st.markdown("### Transactions")
    
    # --------------------------------
    # RISK BADGES
    # --------------------------------
    def risk_badge(level):
    
        if level == "HIGH":
            return '<span class="risk-pill risk-high">HIGH</span>'
    
        elif level == "MEDIUM":
            return '<span class="risk-pill risk-medium">MEDIUM</span>'
    
        return '<span class="risk-pill risk-low">LOW</span>'
    
    
    # --------------------------------
    # DECISION BADGES
    # --------------------------------
    def decision_badge(decision):
    
        if decision == "Auto Approve (AI)":
    
            return '''
            <span class="decision-pill decision-approve">
                Approve (AI)
            </span>
            '''
    
        elif decision == "Manual Review":
    
            return '''
            <span class="decision-pill decision-review">
                Manual Review
            </span>
            '''
    
        return '''
        <span class="decision-pill decision-decline">
            Decline
        </span>
        '''
    display_df = display_df[[
        "Transaction ID",
        "Decision",
        "Risk Level",
        "Risk Score",
        "Expected Cost",
        "Why"
    ]]
    
    # --------------------------------
    # TABLE ROWS
    # --------------------------------
    table_rows = ""
    
    for _, row_data in display_df.iterrows():
    
        table_rows += f"""
        <tr>
    
            <td class="tx-id">
                TX-{int(row_data['Transaction ID']):05d}
            </td>
    
            <td>
                {decision_badge(row_data['Decision'])}
            </td>
    
            <td>
                {risk_badge(row_data['Risk Level'])}
            </td>
    
            <td>
                {row_data['Risk Score']}
            </td>
    
            <td>
                {row_data['Expected Cost']}
            </td>
    
            <td class="reason-cell">
                {row_data['Why']}
            </td>
    
        </tr>
        """
    
    # --------------------------------
    # FINAL TABLE HTML
    # --------------------------------
    table_html = f"""
    
    <div class="ops-table-wrap">
    
        <div class="ops-table-scroll">
    
            <table class="ops-table">
    
                <thead>
                    <tr>
                        <th>Transaction</th>
                        <th>Decision</th>
                        <th>Risk</th>
                        <th>Score</th>
                        <th>Expected Cost</th>
                        <th>Risk Drivers</th>
                    </tr>
                </thead>
    
                <tbody>
                    {table_rows}
                </tbody>
    
            </table>
    
        </div>
    
    </div>
    """
    
    st.markdown(
        table_html,
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    
    if st.button(
        "View Insights →",
        use_container_width=True
    ):
        st.session_state.step = 4
        st.rerun()

# ==============================
# INSIGHTS
# ==============================
def render_insights_page():
    render_topbar(4)

    df = st.session_state.results

    if st.button("← Back"):
        st.session_state.step = 3
        st.rerun()
    
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

    # --------------------------------
    # PREP DISPLAY DATA
    # --------------------------------
    display_df = df.copy().reset_index(drop=True)
    
    display_df["Transaction ID"] = display_df.index + 1
    
    display_df["Decision"] = display_df["optimal_strategy"].apply(map_action)
    
    display_df["Risk Level"] = (
        display_df["risk_probability"]
        .apply(risk_tier)
        .str.upper()
    )
    
    display_df["Risk Score"] = (
        display_df["risk_probability"]
        .map(lambda x: f"{x:.2f}")
    )
    
    display_df["Expected Cost"] = (
        display_df["expected_cost"]
        .map(format_money)
    )
    
    display_df["Why"] = (
        display_df.apply(generate_reason, axis=1)
    )

    st.markdown(f"""
    - {format_money(savings)} saved through optimized decisioning  
    - {automation_rate:.1%} of transactions automated  
    - Manual review focused on highest-risk cases  
    """)

if st.session_state.step == 1:
    render_upload_page()

elif st.session_state.step == 2:
    render_costs_page()

elif st.session_state.step == 3:
    render_decision_page()

elif st.session_state.step == 4:
    render_insights_page()
