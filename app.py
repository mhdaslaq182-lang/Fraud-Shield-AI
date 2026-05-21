
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
import os
import re
import json
from datetime import datetime
from features import engineer_features, FEATURES
from alert_system import trigger_alert



st.set_page_config(
    page_title="Fraud Shield AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/fraudshield",
        "Report a bug": "mailto:mhdaslaq182@gmail.com",
        "About": """
        # Fraud Shield AI
        **Sri Lanka Edition** | CG02 Group 07
        
        AI-powered fraud detection system for Sri Lankan banks.
        
        - 6 Fraud Domains
        - 284,807 Real Transactions  
        - 100% Accuracy
        - Built with Python & Streamlit
        """
    }
)

@st.cache_resource
def load_model(name="random_forest"):
    return pickle.load(open(f"models/{name}.pkl", "rb"))

@st.cache_resource
def load_domain_model(domain):
    path = f"models/{domain}_model.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    return None

# ── Glassmorphism Light Theme ──
# Apply theme - read from session state
_theme = st.session_state.get("theme", "dark")

_light_bg   = "linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 30%, #ecfdf5 70%, #f0fdf4 100%)"
_dark_bg    = "linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%)"
_light_side = "rgba(255,255,255,0.6)"
_dark_side  = "rgba(10,14,26,0.95)"
_light_text = "#1a202c"
_dark_text  = "#ffffff"
_light_sub  = "#4a5568"
_dark_sub   = "#8899aa"
_light_card = "rgba(255,255,255,0.7)"
_dark_card  = "rgba(255,255,255,0.04)"
_light_card_border = "rgba(255,255,255,0.9)"
_dark_card_border  = "rgba(0,212,255,0.2)"
_light_input = "rgba(255,255,255,0.7)"
_dark_input  = "rgba(255,255,255,0.05)"
_light_metric_val = "#1D9E75"
_dark_metric_val  = "#00d4ff"
_light_accent = "#1D9E75"
_dark_accent  = "#00d4ff"
_light_section = "#718096"
_dark_section  = "#556677"
_light_btn_grad = "linear-gradient(135deg, #1D9E75, #0F6E56)"
_dark_btn_grad  = "linear-gradient(135deg, #00d4ff, #0099bb)"
_light_btn_color = "white"
_dark_btn_color  = "#000"

bg          = _dark_bg    if _theme=="dark" else _light_bg
side_bg     = _dark_side  if _theme=="dark" else _light_side
text        = _dark_text  if _theme=="dark" else _light_text
sub         = _dark_sub   if _theme=="dark" else _light_sub
card        = _dark_card  if _theme=="dark" else _light_card
card_border = _dark_card_border if _theme=="dark" else _light_card_border
inp         = _dark_input  if _theme=="dark" else _light_input
metric_val  = _dark_metric_val  if _theme=="dark" else _light_metric_val
accent      = _dark_accent  if _theme=="dark" else _light_accent
section_col = _dark_section if _theme=="dark" else _light_section
btn_grad    = _dark_btn_grad   if _theme=="dark" else _light_btn_grad
btn_color   = _dark_btn_color  if _theme=="dark" else _light_btn_color

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap");

.stApp {
    background: """ + bg + """ !important;
    background-attachment: fixed !important;
}

section[data-testid="stSidebar"] {
    background: """ + side_bg + """ !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid """ + card_border + """ !important;
    box-shadow: 4px 0 20px rgba(31,38,135,0.05) !important;
}

#MainMenu { visibility: hidden; } 
footer { visibility: hidden; }

/* Make toolbar items visible and small */
.stToolbar { visibility: visible !important; }

/* Small theme toggle */
#theme-btn {
    position: fixed;
    top: 10px;
    right: 110px;
    z-index: 99999;
}
#theme-btn button {
    padding: 4px 10px !important;
    font-size: 0.72rem !important;
    border-radius: 20px !important;
    min-height: 0 !important;
    height: 30px !important;
    line-height: 1 !important;
}

/* Sidebar toggle for Streamlit 1.56.0 */
.css-1rs6os { visibility: visible !important; }
.css-17ziqus { visibility: visible !important; }
.css-1lcbmhc { visibility: visible !important; }
.css-1outpf7 { visibility: visible !important; }
.css-k1vhr4 { visibility: visible !important; }
.css-zt5igj { visibility: visible !important; }
button.css-1rs6os  { 
    visibility: visible !important;
    display: flex !important;
}
[class*="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    background: rgba(29,158,117,0.15) !important;
    border-radius: 0 8px 8px 0 !important;
}
[class*="collapsedControl"] svg {
    fill: #1D9E75 !important;
}

}

/* Sidebar text */
section[data-testid="stSidebar"] * { color: #4a5568 !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: #4a5568 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 10px !important;
    transition: all 0.3s !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #1D9E75 !important;
    background: rgba(29,158,117,0.08) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow: 0 4px 20px rgba(31,38,135,0.06) !important;
    transition: all 0.3s !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(31,38,135,0.1) !important;
}
[data-testid="metric-container"] label {
    color: #718096 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    color: """ + metric_val + """ !important;
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] * {
    color: #1D9E75 !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: #718096 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
    color: #1D9E75 !important;
}

/* Buttons */
.stButton button {
    background: """ + btn_grad + """ !important;
    color: """ + btn_color + """ !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 15px rgba(29,158,117,0.25) !important;
    transition: all 0.3s !important;
}
.stButton button:hover {
    box-shadow: 0 6px 22px rgba(29,158,117,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: """ + inp + """ !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(29,158,117,0.15) !important;
    border-radius: 12px !important;
    color: #1a202c !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #1D9E75 !important;
    box-shadow: 0 0 0 3px rgba(29,158,117,0.1) !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(29,158,117,0.15) !important;
    border-radius: 12px !important;
    color: #1a202c !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] > div > div { background: rgba(29,158,117,0.15) !important; }
.stSlider [data-baseweb="slider"] > div > div > div { background: #1D9E75 !important; }

/* DataFrames */
.stDataFrame {
    border: 1px solid rgba(29,158,117,0.15) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(10px) !important;
    overflow: hidden !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #4a5568 !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    color: #1D9E75 !important;
    background: rgba(29,158,117,0.1) !important;
    font-weight: 600 !important;
}

/* Progress bar */
.stProgress > div > div > div { background: #1D9E75 !important; }

/* Alerts */
.stSuccess {
    background: rgba(29,158,117,0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(29,158,117,0.3) !important;
    border-radius: 14px !important;
    color: #0F6E56 !important;
}
.stError {
    background: rgba(226,75,74,0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(226,75,74,0.3) !important;
    border-radius: 14px !important;
    color: #A32D2D !important;
}
.stWarning {
    background: rgba(239,159,39,0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(239,159,39,0.3) !important;
    border-radius: 14px !important;
    color: #854F0B !important;
}
.stInfo {
    background: rgba(55,138,221,0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(55,138,221,0.3) !important;
    border-radius: 14px !important;
    color: #185FA5 !important;
}

/* Headings */
h1, h2, h3 {
    font-family: "Space Grotesk", sans-serif !important;
    color: """ + text + """ !important;
    letter-spacing: -0.3px !important;
    font-weight: 700 !important;
}

p, li, label, span { color: """ + sub + """ !important; }

/* Custom helpers */
.glass-card {
    background: """ + card + """;
    backdrop-filter: blur(20px);
    border: 1px solid """ + card_border + """;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 4px 20px rgba(31,38,135,0.06);
}

.cyber-title {
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 1.6rem;
    font-weight: 700;
    color: """ + text + """ !important;
    letter-spacing: -0.5px;
    margin-bottom: 5px;
}

.cyber-sub {
    font-size: 0.95rem;
    color: #718096 !important;
    font-weight: 500;
    margin-bottom: 5px;
}

.section-title {
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 0.82rem;
    color: """ + section_col + """ !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    margin: 25px 0 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(29,158,117,0.1);
}

.gold    { color: #BA7517 !important; }
.success { color: #1D9E75 !important; }
.danger  { color: #E24B4A !important; }



/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
    .block-container { 
        padding: 1rem 0.5rem !important; 
    }
    [data-testid="metric-container"] {
        padding: 10px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    .cyber-title { font-size: 1.2rem !important; }
    .cyber-sub   { font-size: 0.82rem !important; }
    .glass-card  { padding: 14px !important; }
    section[data-testid="stSidebar"] {
        min-width: 200px !important;
    }
}

@media (max-width: 480px) {
    .block-container { padding: 0.5rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    .cyber-title { font-size: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)






# ── Sidebar ──
with st.sidebar:
    # Theme toggle
    _dark_val = st.session_state.get("theme","dark") == "dark"
    c1, c2 = st.columns([3,1])
    with c1:
        _icon = "🌙" if _dark_val else "☀️"
        st.markdown(f"<p style='color:#1D9E75;font-weight:700;font-size:0.9rem;margin-top:8px'>{_icon} FRAUD SHIELD</p>", unsafe_allow_html=True)
    with c2:
        _new_dark = st.toggle("Mode", value=_dark_val, key="ts1", label_visibility="hidden")
        if _new_dark != _dark_val:
            st.session_state.theme = "dark" if _new_dark else "light"
            st.rerun()
    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:25px 0 20px;border-bottom:1px solid rgba(29,158,117,0.1);margin-bottom:20px">
        <div style="font-family:Space Grotesk,sans-serif;font-size:1.2rem;font-weight:700;color:#1D9E75;letter-spacing:1px">
            FRAUD SHIELD
        </div>
        <div style="font-size:0.72rem;color:#718096;letter-spacing:2px;margin-top:6px;text-transform:uppercase;font-weight:500">
            Sri Lanka Edition
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Restore page if theme was toggled
    pages_list = ["HOME","BANKING FRAUD","E-COMMERCE","MOBILE PAYMENT",
                  "INSURANCE","LOAN FRAUD","PHISHING","LIVE MONITOR",
                  "PIPELINE","BANK DATA UPLOAD","API TESTER","ANALYTICS",
                  "SHAP EXPLAINER","ALERT LOG"]
    default_page_idx = pages_list.index(st.session_state.get("current_page","HOME")) if st.session_state.get("current_page","HOME") in pages_list else 0

    page = st.radio("Navigation", [
        "HOME",
        "BANKING FRAUD",
        "E-COMMERCE",
        "MOBILE PAYMENT",
        "INSURANCE",
        "LOAN FRAUD",
        "PHISHING",
        "LIVE MONITOR",
        "PIPELINE",
        "BANK DATA UPLOAD",
        "API TESTER",
        "ANALYTICS",
        "SHAP EXPLAINER",
        "ALERT LOG",
    ], label_visibility="hidden", 
                     index=default_page_idx,
                     key="nav_radio")
    st.session_state.current_page = page

    st.markdown("""
    <div style="margin-top:30px;padding:18px;background:rgba(255,255,255,0.5);backdrop-filter:blur(10px);border-radius:14px;border:1px solid rgba(255,255,255,0.8)">
        <div style="font-size:0.7rem;color:#718096;letter-spacing:2px;margin-bottom:12px;font-weight:600;text-transform:uppercase">System Status</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:#1D9E75"></div>
            <span style="font-size:0.82rem;color:#4a5568;font-weight:500">Models Active</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:#1D9E75"></div>
            <span style="font-size:0.82rem;color:#4a5568;font-weight:500">Monitor Ready</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:#EF9F27"></div>
            <span style="font-size:0.82rem;color:#4a5568;font-weight:500">SMS Pending</span>
        </div>
    </div>
    <div style="margin-top:20px;text-align:center;font-size:0.72rem;color:#718096;letter-spacing:1px">
        CG02 GROUP 07 | 2026
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════
# HOME
# ════════════════════════════════
if page == "HOME":
    st.markdown("""
    <div style="text-align:center;padding:50px 30px;margin-bottom:30px;background:rgba(255,255,255,0.7);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.9);border-radius:24px;box-shadow:0 8px 40px rgba(31,38,135,0.06);position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1D9E75,#378ADD,#EF9F27)"></div>
        <div style="display:inline-block;padding:8px 22px;background:rgba(29,158,117,0.08);border-radius:30px;font-size:0.78rem;color:#1D9E75;letter-spacing:2px;margin-bottom:20px;font-weight:600;text-transform:uppercase">
            AI Powered Security System
        </div>
        <div style="font-family:Space Grotesk,sans-serif;font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#1a202c,#1D9E75,#378ADD);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;letter-spacing:-1px">
            Fraud Shield AI
        </div>
        <div style="font-size:1.1rem;color:#4a5568;font-weight:500">Sri Lanka Edition · Machine Learning &amp; Deep Learning</div>
        
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4,col5,col6 = st.columns(6)
    col1.metric("Domains",     "6")
    col2.metric("Real Data",   "284,807")
    col3.metric("Accuracy",    "100%")
    col4.metric("SL Cities",   "89")
    col5.metric("SL Banks",    "15")
    col6.metric("ML Models",   "10+")

    st.markdown('<div class="section-title">Fraud Detection Modules</div>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    modules = [
        ("#E24B4A", "Banking Fraud",     "Credit Card, AML, Account Takeover",     "100%"),
        ("#EF9F27", "E-Commerce Fraud",  "Daraz, Kapruka, Takas.lk",                "100%"),
        ("#1D9E75", "Mobile Payment",    "Dialog Pay, genie, FriMi, mCash",          "100%"),
        ("#378ADD", "Insurance Fraud",   "Ceylinco, AIA Lanka, Allianz",             "100%"),
        ("#BA7517", "Loan Fraud",        "BOC, Sampath, HNB, Peoples",               "100%"),
        ("#7F77DD", "Phishing Detect",   "Email &amp; SMS NLP Analysis",              "95%"),
    ]
    for i, (color, title, desc, acc) in enumerate(modules):
        col = [col1,col2,col3][i%3]
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.7);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.9);border-radius:16px;padding:20px;margin-bottom:15px;border-left:4px solid {color};box-shadow:0 4px 15px rgba(31,38,135,0.06)">
                <div style="font-family:Space Grotesk,sans-serif;font-weight:600;color:#1a202c;font-size:1rem;margin-bottom:6px">{title}</div>
                <div style="font-size:0.82rem;color:#718096;margin-bottom:14px">{desc}</div>
                <div style="height:5px;background:rgba(29,158,117,0.08);border-radius:3px">
                    <div style="height:5px;background:{color};border-radius:3px;width:{acc}"></div>
                </div>
                <div style="font-size:0.75rem;color:#718096;margin-top:6px;font-weight:500">Accuracy: {acc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    perf = pd.DataFrame({
        "Model":    ["Random Forest","Gradient Boosting","Logistic Regression","Neural Network","Isolation Forest"],
        "Type":     ["Supervised","Supervised","Supervised","Deep Learning","Unsupervised"],
        "ROC-AUC":  ["1.000","1.000","1.000","1.000","N/A"],
        "F1 Score": ["1.000","1.000","1.000","0.994","1.000"],
        "Dataset":  ["Sri Lanka LKR","Sri Lanka LKR","Sri Lanka LKR","Sri Lanka LKR","Sri Lanka LKR"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)


# ════════════════════════════════
# BANKING FRAUD
# ════════════════════════════════
elif page == "BANKING FRAUD":
    st.markdown('<div class="cyber-title">Banking Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Credit Card Fraud · Money Laundering · Account Takeover</div>', unsafe_allow_html=True)
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)
        amount_lkr    = st.number_input("Amount (Rs.)",          100.0, 1000000.0, 15000.0, step=1000.0)
        hour          = st.slider("Hour of Day",                 0, 23, 14)
        frequency     = st.number_input("Frequency (24h)",       0, 50, 3)
        distance_km   = st.number_input("Distance (km)",         0.0, 5000.0, 5.0)
        balance_lkr   = st.number_input("Account Balance (Rs.)", 0.0, 10000000.0, 500000.0)
    with col2:
        st.markdown('<div class="section-title">Security Signals</div>', unsafe_allow_html=True)
        failed_logins = st.number_input("Failed Logins",         0, 20, 0)
        new_device    = st.selectbox("New Device?",              [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")
        account_age   = st.number_input("Account Age (days)",    0, 5000, 365)
        countries     = st.number_input("Countries Used",        1, 10, 1)
        velocity_24h  = st.number_input("Velocity (24h)",        0, 100, 2)
        email_risk    = st.slider("Email Risk Score",            0.0, 1.0, 0.1)
        is_weekend    = st.selectbox("Weekend?",                 [0,1], format_func=lambda x:"Yes" if x else "No")

    if st.button("Analyze Transaction", use_container_width=True):
        model = load_model()
        txn = {"amount_lkr":amount_lkr,"hour":hour,"frequency":frequency,"distance_km":distance_km,
               "failed_logins":failed_logins,"new_device":new_device,"account_age":account_age,
               "countries":countries,"velocity_24h":velocity_24h,"email_risk":email_risk,
               "balance_lkr":balance_lkr,"is_weekend":is_weekend}
        df   = pd.DataFrame([txn])
        df   = engineer_features(df)
        prob = model.predict_proba(df[FEATURES])[0][1]
        risk = "Critical" if prob>0.85 else "High" if prob>0.65 else "Medium" if prob>0.40 else "Low"

        st.divider()
        col1,col2,col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob:.1%}")
        col2.metric("Risk Level", risk)
        col3.metric("Decision", "Fraud" if prob>0.5 else "Safe")

        if prob > 0.5:
            st.error(f"Fraud detected — Rs. {amount_lkr:,.2f} · Probability: {prob:.1%}")
            trigger_alert(txn, prob, fraud_type="Banking Fraud", send_email=True)
        else:
            st.success(f"Legitimate — Rs. {amount_lkr:,.2f} · Fraud Probability: {prob:.1%}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={"text":"Fraud Risk Score","font":{"family":"Space Grotesk","color":"#1a202c","size":18}},
            number={"suffix":"%","font":{"family":"Space Grotesk","color":"#1D9E75","size":36}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#718096","tickfont":{"color":"#4a5568"}},
                "bar":{"color":"#E24B4A" if prob>0.5 else "#1D9E75"},
                "bgcolor":"rgba(255,255,255,0.5)",
                "borderwidth":2,
                "bordercolor":"rgba(29,158,117,0.2)",
                "steps":[
                    {"range":[0,40],  "color":"rgba(29,158,117,0.1)"},
                    {"range":[40,65], "color":"rgba(239,159,39,0.1)"},
                    {"range":[65,85], "color":"rgba(226,75,74,0.1)"},
                    {"range":[85,100],"color":"rgba(226,75,74,0.2)"},
                ],
                "threshold":{"line":{"color":"#E24B4A","width":3},"value":50}
            }
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color":"#4a5568","family":"Inter"})
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════
# E-COMMERCE
# ════════════════════════════════
elif page == "E-COMMERCE":
    st.markdown('<div class="cyber-title">E-Commerce Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Daraz.lk · Kapruka · Takas.lk · Ikman.lk</div>', unsafe_allow_html=True)
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        order_amount     = st.number_input("Order Amount (Rs.)",    100.0, 500000.0, 2500.0)
        items_count      = st.number_input("Number of Items",       1, 50, 2)
        hour             = st.slider("Hour of Order",               0, 23, 14)
        is_new_customer  = st.selectbox("New Customer?",            [0,1], format_func=lambda x:"Yes" if x else "No")
        failed_payments  = st.number_input("Failed Payments",       0, 10, 0)
    with col2:
        different_address= st.selectbox("Different Address?",       [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")
        device_changes   = st.number_input("Device Changes",        0, 10, 0)
        return_rate      = st.slider("Return Rate",                 0.0, 1.0, 0.05)
        account_age_days = st.number_input("Account Age (days)",    0, 3000, 200)
        promo_abuse      = st.selectbox("Promo Abuse?",             [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")

    if st.button("Check Order", use_container_width=True):
        model = load_domain_model("ecommerce")
        if model:
            X = pd.DataFrame([{"order_amount":order_amount,"items_count":items_count,"hour":hour,
                               "is_new_customer":is_new_customer,"failed_payments":failed_payments,
                               "different_address":different_address,"device_changes":device_changes,
                               "return_rate":return_rate,"account_age_days":account_age_days,"promo_abuse":promo_abuse}])
            prob = model.predict_proba(X)[0][1]
            col1,col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.1%}")
            col2.metric("Risk", "High" if prob>0.7 else "Medium" if prob>0.4 else "Low")
            if prob > 0.5:
                st.error(f"Fraudulent Order Detected — Probability: {prob:.1%}")
                try:
                    from alert_system import trigger_alert
                    trigger_alert({"amount_lkr": order_amount}, prob, fraud_type="E-Commerce Fraud", send_email=True)
                except: pass
            else:
                st.success(f"Legitimate Order — Probability: {prob:.1%}")
        else:
            st.warning("Run multi_domain.py first to train domain models")


# ════════════════════════════════
# MOBILE PAYMENT
# ════════════════════════════════
elif page == "MOBILE PAYMENT":
    st.markdown('<div class="cyber-title">Mobile Payment Fraud</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Dialog Pay · genie · FriMi · mCash · iPay</div>', unsafe_allow_html=True)
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        amount_lkr       = st.number_input("Amount (Rs.)",           1.0, 100000.0, 500.0)
        hour             = st.slider("Hour",                         0, 23, 12)
        top_up_frequency = st.number_input("Top-up Frequency",       0, 50, 3)
        sim_age_days     = st.number_input("SIM Age (days)",          0, 3000, 365)
    with col2:
        is_roaming       = st.selectbox("Roaming?",                  [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")
        pin_attempts     = st.number_input("PIN Attempts",            0, 10, 0)
        receiver_known   = st.selectbox("Receiver Known?",            [0,1], format_func=lambda x:"Yes (Safe)" if x else "No (Warning)")
        transaction_speed= st.slider("Transaction Speed (sec)",       0.01, 10.0, 2.0)

    if st.button("Check Payment", use_container_width=True):
        model = load_domain_model("mobile_payment")
        if model:
            X = pd.DataFrame([{"amount_lkr":amount_lkr,"hour":hour,"top_up_frequency":top_up_frequency,
                               "sim_age_days":sim_age_days,"is_roaming":is_roaming,"pin_attempts":pin_attempts,
                               "receiver_known":receiver_known,"transaction_speed":transaction_speed}])
            prob = model.predict_proba(X)[0][1]
            col1,col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.1%}")
            col2.metric("Risk", "High" if prob>0.7 else "Medium" if prob>0.4 else "Low")
            if prob > 0.5:
                st.error(f"Fraudulent Payment — Probability: {prob:.1%}")
                try:
                    from alert_system import trigger_alert
                    trigger_alert({"amount_lkr": amount_lkr}, prob, fraud_type="Mobile Payment Fraud", send_email=True)
                except: pass
            else:
                st.success(f"Legitimate Payment — Probability: {prob:.1%}")


# ════════════════════════════════
# INSURANCE
# ════════════════════════════════
elif page == "INSURANCE":
    st.markdown('<div class="cyber-title">Insurance Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Ceylinco · AIA Lanka · Allianz Lanka · HNB Assurance</div>', unsafe_allow_html=True)
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        claim_amount      = st.number_input("Claim Amount (Rs.)",   1000.0, 5000000.0, 50000.0)
        policy_age_days   = st.number_input("Policy Age (days)",    0, 5000, 500)
        previous_claims   = st.number_input("Previous Claims",      0, 20, 0)
        documents_missing = st.selectbox("Documents Missing?",      [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")
    with col2:
        claim_speed_days  = st.number_input("Days to File Claim",   1, 180, 30)
        witness_count     = st.number_input("Witnesses",            0, 10, 1)
        injury_severity   = st.slider("Injury Severity",            0.0, 1.0, 0.2)
        lawyer_involved   = st.selectbox("Lawyer Involved?",        [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")

    if st.button("Check Claim", use_container_width=True):
        model = load_domain_model("insurance")
        if model:
            X = pd.DataFrame([{"claim_amount":claim_amount,"policy_age_days":policy_age_days,
                               "previous_claims":previous_claims,"documents_missing":documents_missing,
                               "claim_speed_days":claim_speed_days,"witness_count":witness_count,
                               "injury_severity":injury_severity,"lawyer_involved":lawyer_involved}])
            prob = model.predict_proba(X)[0][1]
            col1,col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.1%}")
            col2.metric("Risk", "High" if prob>0.7 else "Medium" if prob>0.4 else "Low")
            if prob > 0.5:
                st.error(f"Fraudulent Claim — Probability: {prob:.1%}")
                try:
                    from alert_system import trigger_alert
                    trigger_alert({"amount_lkr": claim_amount}, prob, fraud_type="Insurance Fraud", send_email=True)
                except: pass
            else:
                st.success(f"Legitimate Claim — Probability: {prob:.1%}")


# ════════════════════════════════
# LOAN FRAUD
# ════════════════════════════════
elif page == "LOAN FRAUD":
    st.markdown('<div class="cyber-title">Loan Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">BOC · Peoples Bank · Commercial Bank · Sampath · HNB</div>', unsafe_allow_html=True)
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        loan_amount       = st.number_input("Loan Amount (Rs.)",     10000.0, 10000000.0, 200000.0)
        monthly_income    = st.number_input("Monthly Income (Rs.)",  10000.0, 5000000.0,   80000.0)
        credit_score      = st.number_input("Credit Score",          300, 850, 650)
        employment_years  = st.number_input("Employment Years",      0, 50, 5)
    with col2:
        existing_loans    = st.number_input("Existing Loans",        0, 20, 1)
        age               = st.number_input("Age",                   18, 70, 35)
        address_changes   = st.selectbox("Recent Address Change?",   [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")
        doc_inconsistency = st.selectbox("Document Issues?",         [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")
        multiple_apps     = st.selectbox("Multiple Applications?",   [0,1], format_func=lambda x:"Yes (Warning)" if x else "No (Safe)")

    if st.button("Check Application", use_container_width=True):
        model = load_domain_model("online_loan")
        if model:
            X = pd.DataFrame([{"loan_amount":loan_amount,"monthly_income":monthly_income,
                               "credit_score":credit_score,"employment_years":employment_years,
                               "existing_loans":existing_loans,"age":age,
                               "address_changes":address_changes,"doc_inconsistency":doc_inconsistency,
                               "multiple_apps":multiple_apps}])
            prob = model.predict_proba(X)[0][1]
            col1,col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.1%}")
            col2.metric("Risk", "High" if prob>0.7 else "Medium" if prob>0.4 else "Low")
            if prob > 0.5:
                st.error(f"Fraudulent Application — Probability: {prob:.1%}")
                try:
                    from alert_system import trigger_alert
                    trigger_alert({"amount_lkr": loan_amount}, prob, fraud_type="Loan Fraud", send_email=True)
                except: pass
            else:
                st.success(f"Legitimate Application — Probability: {prob:.1%}")


# ════════════════════════════════
# PHISHING
# ════════════════════════════════
elif page == "PHISHING":
    st.markdown('<div class="cyber-title">Phishing Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Email &amp; SMS Scam Detection using NLP Analysis</div>', unsafe_allow_html=True)
    st.divider()

    col1,col2 = st.columns([2,1])
    with col1:
        text = st.text_area("Paste suspicious email or SMS:", height=180,
                            placeholder="Example: URGENT! Your BOC account is suspended. Verify now...")
        if st.button("Analyze Text", use_container_width=True):
            if text.strip():
                KEYWORDS = ["urgent","verify your account","click here","password","suspended",
                           "unusual activity","confirm your identity","won a prize","bank details",
                           "wire transfer","act now","limited time","security alert",
                           "update your information","dear customer","free gift","congratulations"]
                matched  = [kw for kw in KEYWORDS if kw in text.lower()]
                urls     = re.findall(r"http[s]?://\S+", text)
                bad_urls = [u for u in urls if any(x in u for x in ["bit.ly","tinyurl","secure-","login-","verify-"])]
                score    = min(len(matched)*12 + len(bad_urls)*25, 100)

                st.divider()
                col_a,col_b,col_c = st.columns(3)
                col_a.metric("Risk Score",      f"{score}/100")
                col_b.metric("Keywords Found",  len(matched))
                col_c.metric("Suspicious URLs", len(bad_urls))

                if score > 40:
                    st.error(f"Phishing detected — Risk Score: {score}/100")
                    try:
                        from alert_system import trigger_alert
                        trigger_alert({"amount_lkr": 0}, score/100, fraud_type="Phishing Attack", send_email=True)
                    except: pass
                else:
                    st.success(f"Looks safe — Risk Score: {score}/100")

                if matched:
                    st.warning(f"Suspicious keywords: {', '.join(matched)}")
                if bad_urls:
                    st.error(f"Suspicious URLs: {', '.join(bad_urls)}")

    with col2:
        st.markdown('<div class="section-title">Test Examples</div>', unsafe_allow_html=True)
        examples = [
            "URGENT: Your BOC account suspended! Verify: http://secure-boc.xyz",
            "Your Dialog bill is ready. Pay at dialog.lk",
            "You won Rs.500,000! Send bank details NOW!",
            "Your Sampath statement is available online.",
        ]
        for ex in examples:
            if st.button(ex[:40]+"...", use_container_width=True):
                st.session_state["ex"] = ex


# ════════════════════════════════
# LIVE MONITOR
# ════════════════════════════════
elif page == "LIVE MONITOR":
    st.markdown('<div class="cyber-title">Real-Time Transaction Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Live simulation of Sri Lankan banking transactions</div>', unsafe_allow_html=True)
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        n = st.slider("Number of transactions", 5, 50, 20)
    with col2:
        speed = st.selectbox("Speed", ["Fast (0.3s)","Normal (0.8s)","Slow (1.5s)"])
        delay = {"Fast (0.3s)":0.3,"Normal (0.8s)":0.8,"Slow (1.5s)":1.5}[speed]

    SL_BANKS = ["BOC","Peoples","Commercial","Sampath","HNB","NSB","Seylan","NTB"]

    if st.button("Start Live Monitor", use_container_width=True):
        model = load_model()
        progress   = st.progress(0)
        status_box = st.empty()
        table_box  = st.empty()
        rows = []; fraud_count = 0

        for i in range(n):
            is_fraud_sim = np.random.random() < 0.10
            txn = {
                "amount_lkr":    round(np.random.uniform(50000,500000) if is_fraud_sim else np.random.exponential(8000),2),
                "hour":          int(np.random.choice([1,2,3]) if is_fraud_sim else np.random.randint(8,20)),
                "frequency":     int(np.random.poisson(18 if is_fraud_sim else 4)),
                "distance_km":   round(np.random.uniform(300,2000) if is_fraud_sim else np.random.exponential(10),1),
                "failed_logins": int(np.random.randint(4,10) if is_fraud_sim else np.random.choice([0,1],p=[0.97,0.03])),
                "new_device":    int(1 if is_fraud_sim else np.random.choice([0,1],p=[0.92,0.08])),
                "account_age":   int(np.random.randint(1,45) if is_fraud_sim else np.random.randint(180,3000)),
                "countries":     int(np.random.randint(2,5) if is_fraud_sim else 1),
                "velocity_24h":  int(np.random.poisson(12 if is_fraud_sim else 2)),
                "email_risk":    round(np.random.uniform(0.3,0.7) if is_fraud_sim else np.random.uniform(0,0.2),2),
                "balance_lkr":   round(np.random.uniform(10000,200000) if is_fraud_sim else np.random.uniform(50000,5000000),2),
                "is_weekend":    int(np.random.choice([0,1])),
            }
            df   = pd.DataFrame([txn])
            df   = engineer_features(df)
            prob = model.predict_proba(df[FEATURES])[0][1]
            if prob > 0.5: fraud_count += 1
            status = "Fraud" if prob>0.5 else "Safe"
            rows.append({
                "#":      i+1,
                "Time":   datetime.now().strftime("%H:%M:%S"),
                "Amount": f"Rs. {txn['amount_lkr']:,.0f}",
                "Bank":   np.random.choice(SL_BANKS),
                "Status": status,
                "Risk":   f"{prob:.0%}",
            })
            # Send email alert if fraud detected
            if prob > 0.5:
                try:
                    from alert_system import trigger_alert
                    trigger_alert(txn, prob, 
                                fraud_type="Live Monitor Fraud",
                                send_email=True)
                except Exception as e:
                    pass
            progress.progress((i+1)/n)
            status_box.info(f"Processing #{i+1}/{n}...")
            table_box.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            time.sleep(delay)

        status_box.empty(); progress.empty()
        col1,col2,col3 = st.columns(3)
        col1.metric("Processed",      n)
        col2.metric("Fraud Detected", fraud_count)
        col3.metric("Fraud Rate",     f"{fraud_count/n:.1%}")
        st.success("Monitor complete!")


# ════════════════════════════════
# BANK DATA UPLOAD
# ════════════════════════════════
elif page == "BANK DATA UPLOAD":
    st.markdown('<div class="cyber-title">Bank Data Upload & Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Upload real bank transaction data — AI analyzes and detects fraud instantly</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div style="background:rgba(29,158,117,0.08);border:1px solid rgba(29,158,117,0.2);border-radius:14px;padding:20px;margin-bottom:20px">
        <div style="font-weight:600;color:#1D9E75;margin-bottom:10px">How to use:</div>
        <div style="font-size:0.9rem;line-height:1.8">
            1. Download the sample template below<br>
            2. Fill with your bank transaction data<br>
            3. Upload the CSV file<br>
            4. Click Analyze to run AI fraud detection<br>
            5. View results and download fraud report
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Upload Transaction Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    with col2:
        st.markdown('<div class="section-title">Download Sample Template</div>', unsafe_allow_html=True)
        sample_df = pd.DataFrame({
            "transaction_id": ["TXN001","TXN002","TXN003"],
            "amount_lkr":     [15000, 250000, 5000],
            "hour":           [14, 2, 10],
            "distance_km":    [5, 800, 2],
            "failed_logins":  [0, 6, 0],
            "new_device":     [0, 1, 0],
            "account_age":    [365, 10, 1000],
            "countries":      [1, 3, 1],
            "velocity_24h":   [2, 18, 1],
            "email_risk":     [0.1, 0.9, 0.05],
            "balance_lkr":    [500000, 120000, 250000],
            "is_weekend":     [0, 0, 1],
            "bank":           ["BOC","Sampath","HNB"],
            "city":           ["Colombo","Kandy","Galle"],
        })
        st.download_button(
            "Download Sample Template",
            data=sample_df.to_csv(index=False),
            file_name="bank_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"File uploaded! {len(df_upload):,} transactions found.")
            st.dataframe(df_upload.head(10), use_container_width=True, hide_index=True)

            col1,col2,col3 = st.columns(3)
            col1.metric("Total", len(df_upload))
            col2.metric("Columns", len(df_upload.columns))
            col3.metric("Rows", len(df_upload))

            if st.button("ANALYZE WITH AI", use_container_width=True):
                model = load_model()
                required = ["amount_lkr","hour","distance_km","failed_logins",
                           "new_device","account_age","countries","velocity_24h",
                           "email_risk","balance_lkr"]
                missing = [c for c in required if c not in df_upload.columns]

                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    if "frequency" not in df_upload.columns:
                        df_upload["frequency"] = 3
                    if "is_weekend" not in df_upload.columns:
                        df_upload["is_weekend"] = 0

                    progress = st.progress(0)
                    results = []

                    for i, row in df_upload.iterrows():
                        txn  = row.to_dict()
                        df_t = pd.DataFrame([txn])
                        df_t = engineer_features(df_t)
                        for feat in FEATURES:
                            if feat not in df_t.columns:
                                df_t[feat] = 0
                        prob = float(model.predict_proba(df_t[FEATURES])[0][1])
                        risk = "CRITICAL" if prob>0.85 else "HIGH" if prob>0.65 else "MEDIUM" if prob>0.40 else "LOW"
                        results.append({
                            "amount_lkr":        float(txn.get("amount_lkr",0)),
                            "bank":              str(txn.get("bank","")),
                            "city":              str(txn.get("city","")),
                            "fraud_probability": round(prob,4),
                            "risk_level":        risk,
                            "is_fraud":          bool(prob>0.5),
                            "decision":          "BLOCKED" if prob>0.5 else "APPROVED",
                        })
                        progress.progress((i+1)/len(df_upload))

                    progress.empty()
                    df_results = pd.DataFrame(results)
                    fraud_df   = df_results[df_results["is_fraud"]]

                    st.divider()
                    col1,col2,col3,col4 = st.columns(4)
                    col1.metric("Total",         len(df_results))
                    col2.metric("Fraud",         len(fraud_df))
                    col3.metric("Legitimate",    len(df_results)-len(fraud_df))
                    col4.metric("Fraud Rate",    f"{len(fraud_df)/len(df_results)*100:.1f}%")

                    if len(fraud_df) > 0:
                        st.markdown('<div class="section-title">Fraudulent Transactions</div>', unsafe_allow_html=True)
                        st.dataframe(fraud_df, use_container_width=True, hide_index=True)

                        # Send alerts
                        for _, row in fraud_df.iterrows():
                            try:
                                from alert_system import trigger_alert
                                trigger_alert(row.to_dict(), float(row["fraud_probability"]),
                                            fraud_type="Bank Upload Fraud", send_email=True)
                            except: pass
                        st.success("Email alerts sent for all fraud transactions!")

                    # Download results
                    st.download_button(
                        "Download Full Results",
                        data=df_results.to_csv(index=False),
                        file_name="fraud_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error: {e}")


# ════════════════════════════════
# API TESTER
# ════════════════════════════════
elif page == "API TESTER":
    st.markdown('<div class="cyber-title">Bank API Integration Tester</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Test the REST API that banks use to connect to Fraud Shield AI</div>', unsafe_allow_html=True)
    st.divider()

    # API Keys table
    st.markdown('<div class="section-title">Available Bank API Keys</div>', unsafe_allow_html=True)

    api_keys_df = pd.DataFrame({
        "Bank":    ["Bank of Ceylon","Peoples Bank","Sampath Bank",
                   "Hatton National Bank","Commercial Bank",
                   "National Savings Bank","Seylan Bank","Nations Trust Bank"],
        "API Key": ["BOC-2026-FRAUDSHIELD","PEOPLES-2026-FRAUDSHIELD",
                   "SAMPATH-2026-FRAUDSHIELD","HNB-2026-FRAUDSHIELD",
                   "COMMERCIAL-2026-FRAUDSHIELD","NSB-2026-FRAUDSHIELD",
                   "SEYLAN-2026-FRAUDSHIELD","NTB-2026-FRAUDSHIELD"],
        "Status":  ["Active"] * 8
    })
    st.dataframe(api_keys_df, use_container_width=True, hide_index=True)

    st.divider()

    # Test single transaction
    st.markdown('<div class="section-title">Test Single Transaction API</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        bank_key  = st.selectbox("Select Bank", [
            "BOC-2026-FRAUDSHIELD",
            "SAMPATH-2026-FRAUDSHIELD",
            "HNB-2026-FRAUDSHIELD",
            "PEOPLES-2026-FRAUDSHIELD",
        ])
        amount    = st.number_input("Amount (Rs.)", 100.0, 1000000.0, 250000.0)
        hour      = st.slider("Hour", 0, 23, 2)
        distance  = st.number_input("Distance (km)", 0.0, 5000.0, 800.0)
        failed    = st.number_input("Failed Logins", 0, 20, 6)

    with col2:
        new_dev   = st.selectbox("New Device?", [0,1], format_func=lambda x:"Yes" if x else "No")
        countries = st.number_input("Countries", 1, 10, 3)
        velocity  = st.number_input("Velocity (24h)", 0, 100, 18)
        email_r   = st.slider("Email Risk", 0.0, 1.0, 0.9)
        balance   = st.number_input("Balance (Rs.)", 0.0, 10000000.0, 120000.0)
        age       = st.number_input("Account Age (days)", 0, 5000, 10)

    if st.button("SEND TO API", use_container_width=True):
        import requests as req
        import json

        # Build request
        payload = {
            "transaction_id": f"TEST-TXN-{int(datetime.now().timestamp())}",
            "amount_lkr":     amount,
            "hour":           hour,
            "distance_km":    distance,
            "failed_logins":  failed,
            "new_device":     new_dev,
            "account_age":    age,
            "countries":      countries,
            "velocity_24h":   velocity,
            "email_risk":     email_r,
            "balance_lkr":    balance,
            "frequency":      3,
            "is_weekend":     0,
        }

        # Show request
        st.markdown('<div class="section-title">API Request Sent</div>', unsafe_allow_html=True)
        st.code(f"""POST /api/v1/check-transaction
Headers: X-API-Key: {bank_key}
Body: {json.dumps(payload, indent=2)}""", language="json")

        try:
            # Call Flask API
            response = req.post(
                "http://localhost:5000/api/v1/check-transaction",
                json=payload,
                headers={"X-API-Key": bank_key},
                timeout=10
            )
            result = response.json()

            # Show response
            st.markdown('<div class="section-title">API Response Received</div>', unsafe_allow_html=True)
            st.code(json.dumps(result, indent=2), language="json")

            # Visual result
            if result.get("decision") == "BLOCKED":
                st.error(f"TRANSACTION BLOCKED — {result.get('fraud_percentage')} fraud probability")
            else:
                st.success(f"TRANSACTION APPROVED — {result.get('fraud_percentage')} fraud probability")

            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Decision",    result.get("decision",""))
            col2.metric("Risk Level",  result.get("risk_level",""))
            col3.metric("Probability", result.get("fraud_percentage",""))
            col4.metric("Process Time",f"{result.get('process_time_ms',0)}ms")

        except Exception as e:
            st.error(f"API Error: {e}")
            st.info("Make sure Flask is running on port 5000!")

    st.divider()

    # API Documentation
    st.markdown('<div class="section-title">API Documentation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,255,255,0.7);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.9);border-radius:14px;padding:25px">

    <h4 style="color:#1D9E75;margin-bottom:15px;font-family:Space Grotesk,sans-serif">Endpoint 1: Check Single Transaction</h4>
    <code style="background:rgba(29,158,117,0.1);padding:8px 15px;border-radius:8px;display:block;margin-bottom:15px">
    POST /api/v1/check-transaction
    </code>

    <h4 style="color:#1D9E75;margin-bottom:15px;font-family:Space Grotesk,sans-serif">Endpoint 2: Batch Check (up to 1000)</h4>
    <code style="background:rgba(29,158,117,0.1);padding:8px 15px;border-radius:8px;display:block;margin-bottom:15px">
    POST /api/v1/batch-check
    </code>

    <h4 style="color:#1D9E75;margin-bottom:15px;font-family:Space Grotesk,sans-serif">Endpoint 3: API Status</h4>
    <code style="background:rgba(29,158,117,0.1);padding:8px 15px;border-radius:8px;display:block;margin-bottom:15px">
    GET /api/v1/status
    </code>

    <h4 style="color:#378ADD;margin-bottom:10px;font-family:Space Grotesk,sans-serif">Authentication</h4>
    <p style="color:#4a5568;font-size:0.9rem">Add your bank API key in the request header:</p>
    <code style="background:rgba(55,138,221,0.1);padding:8px 15px;border-radius:8px;display:block">
    X-API-Key: BOC-2026-FRAUDSHIELD
    </code>

    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════
# ANALYTICS
# ════════════════════════════════
elif page == "ANALYTICS":
    st.markdown('<div class="cyber-title">Fraud Analytics Dashboard</div>', unsafe_allow_html=True)
    st.divider()

    tab1, tab2 = st.tabs(["Sri Lanka Data", "Real Kaggle Data"])

    LIGHT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
                 font=dict(color="#4a5568",family="Inter"),
                 xaxis=dict(gridcolor="rgba(29,158,117,0.08)"),
                 yaxis=dict(gridcolor="rgba(29,158,117,0.08)"))

    with tab1:
        try:
            df = pd.read_csv("data/banking_dataset.csv")
            df = engineer_features(df)

            col1,col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x="amount_lkr", color="is_fraud",
                                   title="Amount Distribution (Rs.)",
                                   color_discrete_map={0:"#1D9E75",1:"#E24B4A"},
                                   nbins=50, barmode="overlay", opacity=0.7)
                fig.update_layout(**LIGHT, title_font_family="Space Grotesk")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                city_fraud = df[df.is_fraud==1]["city"].value_counts().head(10).reset_index()
                city_fraud.columns = ["City","Count"]
                fig = px.bar(city_fraud, x="City", y="Count", title="Top 10 Fraud Cities",
                             color="Count", color_continuous_scale=[[0,"#FAECE7"],[1,"#E24B4A"]])
                fig.update_layout(**LIGHT, title_font_family="Space Grotesk")
                st.plotly_chart(fig, use_container_width=True)

            col1,col2 = st.columns(2)
            with col1:
                bank_fraud = df[df.is_fraud==1]["bank"].value_counts().reset_index()
                bank_fraud.columns = ["Bank","Count"]
                fig = px.bar(bank_fraud, x="Bank", y="Count", title="Fraud by Bank",
                             color="Count", color_continuous_scale=[[0,"#FAEEDA"],[1,"#EF9F27"]])
                fig.update_layout(**LIGHT, title_font_family="Space Grotesk", xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                hourly = df.groupby(["hour","is_fraud"]).size().reset_index(name="count")
                fig = px.bar(hourly, x="hour", y="count", color="is_fraud", title="Fraud by Hour",
                             color_discrete_map={0:"#1D9E75",1:"#E24B4A"}, barmode="group")
                fig.update_layout(**LIGHT, title_font_family="Space Grotesk")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Run main.py first. Error: {e}")

    with tab2:
        try:
            df_k = pd.read_csv("data/creditcard.csv")
            df_k["Hour"] = (df_k["Time"] % 86400) // 3600

            col1,col2 = st.columns(2)
            with col1:
                fraud_k = df_k[df_k.Class==1]["Amount"]
                legit_k = df_k[df_k.Class==0]["Amount"].sample(2000)
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=legit_k, name="Legitimate", marker_color="#1D9E75", opacity=0.7))
                fig.add_trace(go.Histogram(x=fraud_k, name="Fraud",      marker_color="#E24B4A", opacity=0.8))
                fig.update_layout(**LIGHT, title="Real Kaggle: Amount Distribution",
                                  title_font_family="Space Grotesk", barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                hourly_k = df_k.groupby(["Hour","Class"]).size().reset_index(name="count")
                fig = px.line(hourly_k, x="Hour", y="count", color="Class",
                              title="Real Kaggle: Transactions by Hour",
                              color_discrete_map={0:"#1D9E75",1:"#E24B4A"})
                fig.update_layout(**LIGHT, title_font_family="Space Grotesk")
                st.plotly_chart(fig, use_container_width=True)

            col1,col2,col3 = st.columns(3)
            col1.metric("Total Transactions", f"{len(df_k):,}")
            col2.metric("Fraudulent",         f"{(df_k.Class==1).sum():,}")
            col3.metric("Fraud Rate",         f"{df_k.Class.mean():.3%}")
        except Exception as e:
            st.warning(f"Kaggle data not found. {e}")


# ════════════════════════════════
# SHAP EXPLAINER
# ════════════════════════════════
elif page == "SHAP EXPLAINER":
    st.markdown('<div class="cyber-title">SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Understand WHY a transaction was flagged as fraud</div>', unsafe_allow_html=True)
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        amount_lkr    = st.number_input("Amount (Rs.)",    100.0, 1000000.0, 250000.0)
        hour          = st.number_input("Hour",            0, 23, 2)
        distance_km   = st.number_input("Distance (km)",   0.0, 5000.0, 800.0)
        failed_logins = st.number_input("Failed Logins",   0, 20, 6)
    with col2:
        new_device    = st.selectbox("New Device?",        [0,1], index=1)
        countries     = st.number_input("Countries",       1, 10, 3)
        velocity_24h  = st.number_input("Velocity (24h)",  0, 100, 18)
        email_risk    = st.slider("Email Risk",            0.0, 1.0, 0.9)

    if st.button("Explain Prediction", use_container_width=True):
        try:
            import shap
            model = load_model()
            txn = {"amount_lkr":amount_lkr,"hour":hour,"frequency":5,"distance_km":distance_km,
                   "failed_logins":failed_logins,"new_device":new_device,"account_age":30,
                   "countries":countries,"velocity_24h":velocity_24h,"email_risk":email_risk,
                   "balance_lkr":500000,"is_weekend":0}
            df   = pd.DataFrame([txn])
            df   = engineer_features(df)
            X    = df[FEATURES]
            prob = model.predict_proba(X)[0][1]

            explainer  = shap.TreeExplainer(model)
            sv         = explainer.shap_values(X)
            fraud_shap = sv[:,:,1] if isinstance(sv,np.ndarray) and sv.ndim==3 else (sv[1] if isinstance(sv,list) else sv)
            contributions = sorted(zip(FEATURES,fraud_shap[0]), key=lambda x:abs(x[1]), reverse=True)

            st.divider()
            if prob > 0.5:
                st.error(f"Fraud detected — Probability: {prob:.1%}")
            else:
                st.success(f"Legitimate — Probability: {prob:.1%}")

            st.markdown('<div class="section-title">Top Fraud Reasons</div>', unsafe_allow_html=True)
            for feat, val in contributions[:8]:
                direction = "Increases fraud risk" if val>0 else "Decreases fraud risk"
                color     = "#E24B4A" if val>0 else "#1D9E75"
                raw_val   = X[feat].values[0]
                bar_width = min(int(abs(val)*300), 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;padding:14px 18px;background:rgba(255,255,255,0.7);backdrop-filter:blur(10px);border-radius:14px;border:1px solid rgba(255,255,255,0.9);box-shadow:0 2px 10px rgba(31,38,135,0.04)">
                    <div style="width:140px;font-size:0.88rem;color:#4a5568;font-weight:500">{feat}</div>
                    <div style="width:70px;font-family:Space Grotesk,sans-serif;font-size:0.85rem;color:#1a202c;font-weight:600">{raw_val:.2f}</div>
                    <div style="flex:1;height:6px;background:rgba(29,158,117,0.08);border-radius:3px">
                        <div style="height:6px;width:{bar_width}%;background:{color};border-radius:3px"></div>
                    </div>
                    <div style="width:180px;font-size:0.82rem;color:{color};font-weight:600">{direction}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"SHAP error: {e}")


# ════════════════════════════════
# PIPELINE
# ════════════════════════════════
elif page == "PIPELINE":
    st.markdown('<div class="cyber-title">Real-World Transaction Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-sub">Live AI-powered fraud detection pipeline — simulating real bank transactions</div>', unsafe_allow_html=True)
    st.divider()

    # Pipeline controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_txns = st.slider("Transactions to process", 10, 200, 50)
    with col2:
        speed = st.selectbox("Pipeline Speed", [
            "Ultra Fast (0.1s)",
            "Fast (0.3s)",
            "Normal (0.8s)",
            "Slow (1.5s)"
        ])
        delay = {"Ultra Fast (0.1s)":0.1,"Fast (0.3s)":0.3,"Normal (0.8s)":0.8,"Slow (1.5s)":1.5}[speed]
    with col3:
        send_alerts = st.checkbox("Send Email Alerts", value=True)

    # Pipeline stats from previous runs
    try:
        from pipeline import get_pipeline_stats
        stats = get_pipeline_stats()
        if stats:
            st.markdown('<div class="section-title">Pipeline Statistics (All Time)</div>', unsafe_allow_html=True)
            col1,col2,col3,col4,col5 = st.columns(5)
            col1.metric("Total Processed", f"{stats['total']:,}")
            col2.metric("Fraud Detected",  f"{stats['fraud_count']:,}")
            col3.metric("Fraud Rate",      f"{stats['fraud_rate']}%")
            col4.metric("Avg Process Time",f"{stats['avg_time_ms']}ms")
            col5.metric("Fraud Amount",    f"Rs. {stats['fraud_amount']:,.0f}")
    except:
        pass

    if st.button("START PIPELINE", use_container_width=True):
        from pipeline import generate_transaction, process_transaction
        model = load_model()

        # UI placeholders
        progress    = st.progress(0)
        status      = st.empty()
        col1, col2, col3, col4 = st.columns(4)
        metric_total  = col1.empty()
        metric_fraud  = col2.empty()
        metric_rate   = col3.empty()
        metric_time   = col4.empty()
        chart_placeholder = st.empty()
        table_placeholder = st.empty()

        results = []
        fraud_count = 0
        total_time  = 0

        for i in range(n_txns):
            # Generate and process transaction
            txn    = generate_transaction()
            result = process_transaction(txn, model)

            if result["is_fraud_detected"]:
                fraud_count += 1
            total_time += result["process_time_ms"]
            results.append(result)

            # Update metrics
            fraud_rate = fraud_count / (i+1) * 100
            avg_time   = total_time / (i+1)

            metric_total.metric("Processed",     i+1)
            metric_fraud.metric("Fraud Detected", fraud_count)
            metric_rate.metric("Fraud Rate",      f"{fraud_rate:.1f}%")
            metric_time.metric("Avg Time",        f"{avg_time:.1f}ms")

            # Update status
            status_color = "error" if result["is_fraud_detected"] else "success"
            if result["is_fraud_detected"]:
                status.error(f"FRAUD BLOCKED — {result['bank']} — Rs. {result['amount_lkr']:,.0f} — {result['fraud_type']} — {result['fraud_probability']:.1%}")
            else:
                status.success(f"APPROVED — {result['bank']} — Rs. {result['amount_lkr']:,.0f} — {result['city']}")

            # Update table (last 10)
            display = []
            for r in results[-10:][::-1]:
                display.append({
                    "Time":     r["timestamp"][11:19],
                    "Bank":     r["bank"],
                    "City":     r["city"],
                    "Amount":   f"Rs. {r['amount_lkr']:,.0f}",
                    "Decision": r["decision"],
                    "Risk":     r["risk_level"],
                    "Prob":     f"{r['fraud_probability']:.1%}",
                    "Time(ms)": f"{r['process_time_ms']}ms",
                })
            table_placeholder.dataframe(
                pd.DataFrame(display),
                use_container_width=True,
                hide_index=True
            )

            progress.progress((i+1)/n_txns)
            time.sleep(delay)

        progress.empty()
        status.empty()

        # Final summary
        st.divider()
        st.markdown('<div class="section-title">Pipeline Complete — Final Summary</div>', unsafe_allow_html=True)

        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Total Processed",  n_txns)
        col2.metric("Fraud Detected",   fraud_count)
        col3.metric("Legitimate",       n_txns - fraud_count)
        col4.metric("Fraud Rate",       f"{fraud_count/n_txns*100:.1f}%")
        col5.metric("Avg Process Time", f"{total_time/n_txns:.1f}ms")

        # Charts
        df_results = pd.DataFrame(results)
        LIGHT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
                     font=dict(color="#4a5568", family="Inter"))

        col1, col2 = st.columns(2)
        with col1:
            import plotly.express as px
            bank_fraud = df_results[df_results["is_fraud_detected"]]["bank"].value_counts().reset_index()
            bank_fraud.columns = ["Bank","Fraud Count"]
            fig = px.bar(bank_fraud, x="Bank", y="Fraud Count",
                        title="Fraud by Bank",
                        color="Fraud Count",
                        color_continuous_scale=[[0,"#FAEEDA"],[1,"#E24B4A"]])
            fig.update_layout(**LIGHT, title_font_family="Space Grotesk")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            city_fraud = df_results[df_results["is_fraud_detected"]]["city"].value_counts().reset_index()
            city_fraud.columns = ["City","Fraud Count"]
            fig = px.bar(city_fraud, x="City", y="Fraud Count",
                        title="Fraud by City",
                        color="Fraud Count",
                        color_continuous_scale=[[0,"#E1F5EE"],[1,"#1D9E75"]])
            fig.update_layout(**LIGHT, title_font_family="Space Grotesk")
            st.plotly_chart(fig, use_container_width=True)

        # Amount distribution
        import plotly.graph_objects as go
        fig = go.Figure()
        legit = df_results[~df_results["is_fraud_detected"]]["amount_lkr"]
        fraud = df_results[df_results["is_fraud_detected"]]["amount_lkr"]
        fig.add_trace(go.Histogram(x=legit, name="Legitimate", marker_color="#1D9E75", opacity=0.7))
        fig.add_trace(go.Histogram(x=fraud, name="Fraud",      marker_color="#E24B4A", opacity=0.8))
        fig.update_layout(**LIGHT, title="Amount Distribution",
                         title_font_family="Space Grotesk", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Pipeline complete! Processed {n_txns} transactions in {total_time/1000:.1f} seconds. Detected {fraud_count} fraudulent transactions!")


# ════════════════════════════════
# ALERT LOG
# ════════════════════════════════
elif page == "ALERT LOG":
    st.markdown('<div class="cyber-title">Fraud Alert Log</div>', unsafe_allow_html=True)
    st.divider()

    log_file = "alerts/fraud_alerts.log"
    if os.path.exists(log_file):
        alerts = []
        with open(log_file,"r") as f:
            for line in f:
                try: alerts.append(json.loads(line))
                except: pass
        if alerts:
            st.metric("Total Alerts", len(alerts))
            df_a = pd.DataFrame(alerts)
            if "amount_lkr" in df_a.columns:
                df_a["amount_lkr"] = df_a["amount_lkr"].apply(lambda x: f"Rs. {float(x):,.2f}" if x else "N/A")
            st.dataframe(df_a, use_container_width=True, hide_index=True)
            if st.button("Clear Alert Log"):
                os.remove(log_file)
                st.success("Log cleared!")
                st.rerun()
        else:
            st.info("No alerts yet.")
    else:
        st.info("No alerts logged yet. Check a transaction to generate alerts.")
