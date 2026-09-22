import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import sys
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. APPLICATION SETUP & PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ClaimCheck AI — Smart Claim Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================================================================
# 2. TASK-5 ML MODEL SPECIFICATIONS & PREPROCESSING
# ==============================================================================
# Exact 29 feature sequence from Task-5.ipynb
FEATURE_ORDER = [
    'age_of_driver',
    'gender',
    'marital_status',
    'safety_rating',
    'annual_income',
    'high_education',
    'address_change',
    'property_status',
    'zip_code',
    'claim_day_of_week',
    'accident_site',
    'past_num_of_claims',
    'witness_present',
    'liab_prct',
    'channel',
    'police_report',
    'age_of_vehicle',
    'vehicle_category',
    'vehicle_price',
    'vehicle_color',
    'total_claim',
    'injury_claim',
    'policy deductible',
    'annual premium',
    'days open',
    'form defects',
    'claim_day',
    'claim_month',
    'claim_year'
]

# Baseline dataset defaults (median/mode from Task-5)
FEATURE_DEFAULTS = {
    'age_of_driver': 0.40,
    'gender': 1,
    'marital_status': 1,
    'safety_rating': 0.58,
    'annual_income': 0.45,
    'high_education': 1,
    'address_change': 0,
    'property_status': 0,
    'zip_code': 50000,
    'claim_day_of_week': 3,
    'accident_site': 1,
    'past_num_of_claims': 0.10,
    'witness_present': 0,
    'liab_prct': 0.50,
    'channel': 1,
    'police_report': 0,
    'age_of_vehicle': 0.47,
    'vehicle_category': 1,
    'vehicle_color': 3,
    'total_claim': 0.48,
    'injury_claim': 0.46,
    'policy deductible': 0.40,
    'annual premium': 0.53,
    'days open': 0.35,
    'form defects': 4,
    'claim_day': 15,
    'claim_month': 6,
    'claim_year': 2024
}

# The 5 algorithms from Task-5.ipynb with simple descriptions
ALGORITHMS_INFO = {
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest",
        "icon": "🌲",
        "friendly_description": "Combines several decision-making paths together to find a more reliable answer.",
        "test_accuracy": "78.44%",
        "model_file": "random_forest.pkl",
        "requires_scaler": False
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "Decision Tree",
        "icon": "🌳",
        "friendly_description": "Follows a series of simple questions step-by-step to reach a clear answer.",
        "test_accuracy": "78.27%",
        "model_file": "decision_tree.pkl",
        "requires_scaler": False
    },
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "icon": "📊",
        "friendly_description": "Estimates the likelihood of each outcome based on typical claim patterns.",
        "test_accuracy": "78.07%",
        "model_file": "logistic_regression.pkl",
        "requires_scaler": False
    },
    "adaboost": {
        "id": "adaboost",
        "name": "AdaBoost",
        "icon": "⚡",
        "friendly_description": "Focuses closely on difficult or unusual cases to make an accurate check.",
        "test_accuracy": "78.40%",
        "model_file": "adaboost.pkl",
        "requires_scaler": False
    },
    "knn": {
        "id": "knn",
        "name": "K-Nearest Neighbors (KNN)",
        "icon": "👥",
        "friendly_description": "Compares your claim with similar past claims to see what happened.",
        "test_accuracy": "70.98%",
        "model_file": "knn_model.pkl",
        "scaler_file": "knn_scaler.pkl",
        "requires_scaler": True
    }
}

@st.cache_resource
def load_cached_model(algo_id: str):
    """Loads a pre-trained model directly from models/ with in-memory caching."""
    meta = ALGORITHMS_INFO.get(algo_id, ALGORITHMS_INFO["random_forest"])
    file_path = os.path.join(MODELS_DIR, meta["model_file"])
    if not os.path.exists(file_path):
        # Fallback to fraud_model.pkl
        file_path = os.path.join(MODELS_DIR, "fraud_model.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_cached_scaler():
    """Loads the StandardScaler fitted in Task-5 for KNN."""
    scaler_path = os.path.join(MODELS_DIR, "knn_scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

def convert_to_normalized_features(user_inputs: dict) -> pd.DataFrame:
    """
    Translates normal human-friendly inputs (e.g. ₹50,000, 35 yrs, 45 days, 4 stars)
    into the internal normalized 29-feature schema from Task-5.
    """
    norm = dict(user_inputs)

    # Driver Age (18 to 74 years in dataset)
    if 'age_of_driver' in norm and float(norm['age_of_driver']) > 1.0:
        norm['age_of_driver'] = max(0.0, min(1.0, (float(norm['age_of_driver']) - 18.0) / 56.0))

    # Annual Income (₹0 to ₹100,000 baseline)
    if 'annual_income' in norm and float(norm['annual_income']) > 1.0:
        norm['annual_income'] = max(0.0, min(1.0, float(norm['annual_income']) / 100000.0))

    # Vehicle Price (₹5,000 to ₹100,000 baseline)
    if 'vehicle_price' in norm and float(norm['vehicle_price']) > 1.0:
        norm['vehicle_price'] = max(0.0, min(1.0, float(norm['vehicle_price']) / 100000.0))

    # Total Claim Amount (₹1,000 to ₹100,000 baseline)
    if 'total_claim' in norm and float(norm['total_claim']) > 1.0:
        norm['total_claim'] = max(0.0, min(1.0, float(norm['total_claim']) / 100000.0))

    # Injury Claim Amount (₹0 to ₹50,000 baseline)
    if 'injury_claim' in norm and float(norm['injury_claim']) > 1.0:
        norm['injury_claim'] = max(0.0, min(1.0, float(norm['injury_claim']) / 50000.0))

    # Days Claim Open (1 to 365 days)
    if 'days_open' in norm and float(norm['days_open']) > 1.0:
        norm['days open'] = max(0.0, min(1.0, float(norm['days_open']) / 365.0))
    elif 'days open' in norm and float(norm['days open']) > 1.0:
        norm['days open'] = max(0.0, min(1.0, float(norm['days open']) / 365.0))

    # Safety Rating (1 to 5 stars: 1 star = 0.0, 5 stars = 1.0)
    if 'safety_rating' in norm:
        sr = float(norm['safety_rating'])
        if sr >= 1.0 and sr <= 5.0:
            norm['safety_rating'] = max(0.0, min(1.0, (sr - 1.0) / 4.0))

    # Construct DataFrame in exact 29-feature order
    row = {}
    for feature in FEATURE_ORDER:
        if feature in norm:
            row[feature] = norm[feature]
        else:
            alias_map = {
                'past_claims': 'past_num_of_claims',
                'policy_deductible': 'policy deductible',
                'annual_premium': 'annual premium',
                'days_open': 'days open',
                'form_defects': 'form defects'
            }
            mapped = None
            for ak, orig in alias_map.items():
                if orig == feature and ak in norm:
                    mapped = norm[ak]
                    break
            row[feature] = mapped if mapped is not None else FEATURE_DEFAULTS.get(feature, 0.0)

    return pd.DataFrame([row], columns=FEATURE_ORDER)

def execute_claim_check(algo_id: str, inputs: dict):
    """
    Executes in-memory prediction using trained .pkl model without retraining.
    Faithfully maps model classes, probabilities, and decision thresholds.
    """
    meta = ALGORITHMS_INFO.get(algo_id, ALGORITHMS_INFO["random_forest"])
    model = load_cached_model(algo_id)
    features_df = convert_to_normalized_features(inputs)

    # Apply StandardScaler if KNN (matching Task-5 cell 36)
    if meta.get("requires_scaler", False):
        scaler = load_cached_scaler()
        processed_input = scaler.transform(features_df)
    else:
        processed_input = features_df

    # In-memory inference
    pred_raw = model.predict(processed_input)
    prediction_code = int(pred_raw[0])

    # Dynamic class lookup from model.classes_
    classes = list(getattr(model, "classes_", [0, 1]))
    fraud_idx = classes.index(1) if 1 in classes else 1
    normal_idx = classes.index(0) if 0 in classes else 0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_input)[0]
        prob_fraud = float(probs[fraud_idx] * 100.0)
        prob_normal = float(probs[normal_idx] * 100.0)
    elif hasattr(model, "decision_function"):
        df_val = float(model.decision_function(processed_input)[0])
        prob_fraud = float(1.0 / (1.0 + np.exp(-df_val)) * 100.0)
        prob_normal = 100.0 - prob_fraud
    else:
        prob_fraud = 85.0 if prediction_code == 1 else 15.0
        prob_normal = 100.0 - prob_fraud

    is_fraud = (prediction_code == 1)

    if is_fraud:
        friendly_title = "This claim may require further review"
        friendly_explanation = "Based on the trained machine learning model, this claim exhibits patterns that deviate from normal claims and may require additional verification or documentation."
        risk_status = "Review Recommended"
    else:
        friendly_title = "This claim appears normal"
        friendly_explanation = "Based on the trained machine learning model, this claim follows standard patterns consistent with typical legitimate claims."
        risk_status = "Normal"

    # Factual information considered (truthful, non-fabricated)
    raw_income = inputs.get('annual_income', 0)
    income_str = "₹ 0 (Unemployed / Unreported)" if raw_income == 0 else f"₹ {int(raw_income):,}"
    
    considered_factors = [
        {"factor": "Driver Annual Income", "value": income_str, "note": "Primary predictive feature in tree models (56%–87% feature importance)"},
        {"factor": "Total Claim Amount", "value": f"₹ {int(inputs.get('total_claim', 0)):,}", "note": "Total financial amount requested on the claim"},
        {"factor": "Injury Claim Portion", "value": f"₹ {int(inputs.get('injury_claim', 0)):,}", "note": "Portion designated for medical or bodily injury"},
        {"factor": "Days Claim Open", "value": f"{int(inputs.get('days_open', 0))} days", "note": "Elapsed duration since the incident claim was initiated"},
        {"factor": "Official Police Report", "value": "Filed (Yes)" if int(features_df['police_report'].iloc[0]) == 1 else "Not Filed (No)", "note": "Official law enforcement accident documentation"},
        {"factor": "Vehicle Safety Rating", "value": inputs.get('safety_stars_label', 'Standard'), "note": "Vehicle crash test and safety assessment"},
        {"factor": "Driver Past Claims", "value": inputs.get('past_claims_label', 'None'), "note": "Historical record of previous insurance claims"}
    ]

    return {
        "algorithm": meta["name"],
        "algorithm_id": algo_id,
        "is_fraud": is_fraud,
        "prediction_code": prediction_code,
        "friendly_title": friendly_title,
        "friendly_explanation": friendly_explanation,
        "probability_fraud": round(prob_fraud, 1),
        "probability_normal": round(prob_normal, 1),
        "probability": round(prob_fraud, 1),
        "risk_status": risk_status,
        "considered_factors": considered_factors,
        "timestamp": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "diagnostics": {
            "model_file": meta["model_file"],
            "model_class": str(type(model).__name__),
            "model_classes": [int(c) for c in classes],
            "raw_prediction": prediction_code,
            "decision_threshold": "50.0%",
            "class_probabilities": {
                "0 (Normal)": round(prob_normal / 100.0, 4),
                "1 (Fraud)": round(prob_fraud / 100.0, 4)
            },
            "input_shape": list(getattr(processed_input, "shape", (1, 29))),
            "feature_names": FEATURE_ORDER,
            "accuracy": meta["test_accuracy"],
            "requires_scaler": meta["requires_scaler"],
            "internal_features": {col: round(float(features_df[col].iloc[0]), 4) for col in FEATURE_ORDER}
        }
    }

# ==============================================================================
# 3. PREDICTION HISTORY PERSISTENCE
# ==============================================================================
def read_history_records() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_history_record(record: dict):
    records = read_history_records()
    new_entry = {
        "id": datetime.now().strftime("%f")[:8],
        **record,
        "timestamp": datetime.now().strftime("%d %B %Y, %I:%M %p")
    }
    records.insert(0, new_entry)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        return True
    except Exception:
        return False

def delete_history_record(rec_id: str):
    records = read_history_records()
    records = [r for r in records if str(r.get("id")) != str(rec_id)]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        return True
    except Exception:
        return False

def clear_all_records():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return True
    except Exception:
        return False

# ==============================================================================
# 4. MODERN LIGHT THEME CSS INJECTION
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #F8FAFC;
        --surface: #FFFFFF;
        --primary: #4F46E5;
        --primary-light: #EEF2FF;
        --accent-mint: #10B981;
        --accent-coral: #F43F5E;
        --text-dark: #0F172A;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: var(--bg-main);
        color: var(--text-dark);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1360px;
    }

    .brand-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        padding: 0.9rem 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .brand-logo-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.4rem;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.22);
    }
    .brand-name {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .brand-tagline {
        font-size: 0.825rem;
        color: #64748B;
        font-weight: 500;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .pulse-dot-green {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        animation: pulseDot 1.8s infinite;
    }
    @keyframes pulseDot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .step-indicator-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        padding: 0.85rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: #94A3B8;
        transition: all 0.2s ease;
    }
    .step-num {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        background: #F1F5F9;
        color: #64748B;
    }
    .step-active {
        color: #4F46E5 !important;
    }
    .step-active .step-num {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 10px rgba(79, 70, 229, 0.35);
    }
    .step-completed {
        color: #059669 !important;
    }
    .step-completed .step-num {
        background: #ECFDF5 !important;
        color: #059669 !important;
        border: 1px solid #A7F3D0;
    }
    .step-arrow {
        color: #CBD5E1;
        font-weight: 400;
        font-size: 1.1rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F3FF 50%, #EFF6FF 100%);
        border: 1px solid #E0E7FF;
        border-radius: 20px;
        padding: 2.75rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(79, 70, 229, 0.06);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.785rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 0.85rem;
        letter-spacing: -0.025em;
    }
    .hero-highlight {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        max-width: 740px;
        margin-bottom: 1.5rem;
    }

    .algo-grid-card {
        background: #FFFFFF;
        border: 1.5px solid var(--border-color);
        border-radius: 16px;
        padding: 1.35rem 1.25rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .algo-grid-card:hover {
        border-color: #818CF8;
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -6px rgba(79, 70, 229, 0.12);
    }
    .algo-card-active {
        border: 2px solid #4F46E5 !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F3FF 100%) !important;
        box-shadow: 0 12px 28px -4px rgba(79, 70, 229, 0.2) !important;
    }
    .algo-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.85rem;
    }
    .algo-icon-box {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: #F1F5F9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.45rem;
    }
    .algo-name {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }
    .algo-desc {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    .result-banner-safe {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 6px solid #059669;
        border-top: 1px solid #6EE7B7;
        border-right: 1px solid #6EE7B7;
        border-bottom: 1px solid #6EE7B7;
        border-radius: 14px;
        padding: 1.4rem 1.75rem;
        margin-bottom: 1.5rem;
        color: #065F46;
    }
    .result-banner-review {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 6px solid #DC2626;
        border-top: 1px solid #FCA5A5;
        border-right: 1px solid #FCA5A5;
        border-bottom: 1px solid #FCA5A5;
        border-radius: 14px;
        padding: 1.4rem 1.75rem;
        margin-bottom: 1.5rem;
        color: #991B1B;
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .result-subtitle {
        font-size: 0.92rem;
        opacity: 0.95;
        line-height: 1.5;
    }

    .factor-row {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .factor-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    .factor-val {
        font-size: 0.825rem;
        color: #64748B;
    }
    .badge-verified { background: #ECFDF5; color: #059669; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 6px; }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        display: flex;
        flex-direction: column;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
    }

    .history-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        padding: 0.65rem 1.35rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. BRAND NAVIGATION & TABS
# ==============================================================================
st.markdown("""
<div class="brand-navbar">
    <div class="brand-title-wrap">
        <div class="brand-logo-icon">🛡️</div>
        <div>
            <div class="brand-name">ClaimCheck AI <span style="font-size: 0.8rem; font-weight: 500; color: #6366F1; margin-left: 0.35rem;">Task-5 Edition</span></div>
            <div class="brand-tagline">Fast, simple and reliable verification for insurance claims</div>
        </div>
    </div>
    <div class="status-badge">
        <div class="pulse-dot-green"></div>
        Ready to Check Claims
    </div>
</div>
""", unsafe_allow_html=True)

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Home"
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1
if "selected_algo_id" not in st.session_state:
    st.session_state["selected_algo_id"] = "random_forest"
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_human_inputs" not in st.session_state:
    st.session_state["last_human_inputs"] = None
if "is_saved" not in st.session_state:
    st.session_state["is_saved"] = False

# Navigation Tabs
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1.2, 1.1, 2.5])
with nav_col1:
    if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state["active_tab"] == "Home" else "secondary"):
        st.session_state["active_tab"] = "Home"
        st.rerun()
with nav_col2:
    if st.button("⚡ Check a Claim", use_container_width=True, type="primary" if st.session_state["active_tab"] == "Assess" else "secondary"):
        st.session_state["active_tab"] = "Assess"
        st.rerun()
with nav_col3:
    if st.button("📜 Previous Checks", use_container_width=True, type="primary" if st.session_state["active_tab"] == "History" else "secondary"):
        st.session_state["active_tab"] = "History"
        st.rerun()
with nav_col4:
    st.write("")

# ==============================================================================
# TAB 1: HOME & OVERVIEW
# ==============================================================================
if st.session_state["active_tab"] == "Home":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge">Smart Verification System &middot; Task-5</div>
        <div class="hero-title">Turn Data Into <span class="hero-highlight">Predictions.</span></div>
        <div class="hero-desc">
            Choose your machine-learning algorithm, enter the required details, and generate a prediction using your trained ML models. 
            Designed so that anyone can quickly check an insurance claim without needing any technical knowledge.
        </div>
        <div style="display: flex; gap: 1.2rem; flex-wrap: wrap; margin-top: 1rem;">
            <div style="background: rgba(255,255,255,0.7); padding: 0.6rem 1.1rem; border-radius: 10px; border: 1px solid #E2E8F0; font-size: 0.85rem; font-weight: 600; color: #334155;">
                🌲 5 Verification Methods from Task-5
            </div>
            <div style="background: rgba(255,255,255,0.7); padding: 0.6rem 1.1rem; border-radius: 10px; border: 1px solid #E2E8F0; font-size: 0.85rem; font-weight: 600; color: #334155;">
                ⚡ Instant Clear Results
            </div>
            <div style="background: rgba(255,255,255,0.7); padding: 0.6rem 1.1rem; border-radius: 10px; border: 1px solid #E2E8F0; font-size: 0.85rem; font-weight: 600; color: #334155;">
                💾 Save & Download History
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("### 💡 Ways We Can Check Your Claim")
        st.caption("Each method uses a slightly different approach to review your claim details.")
        for k, algo in ALGORITHMS_INFO.items():
            with st.container():
                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                        <span style="font-size: 1.6rem;">{algo['icon']}</span>
                        <div>
                            <div style="font-size: 0.98rem; font-weight: 700; color: #0F172A;">{algo['name']}</div>
                            <div style="font-size: 0.825rem; color: #475569;">{algo['friendly_description']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with c2:
        st.markdown("### 🚀 Ready to Check a Claim?")
        st.caption("Takes less than a minute to complete.")
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
            <div style="font-size: 0.9rem; font-weight: 700; color: #334155; margin-bottom: 0.75rem;">Simple 4-Step Process:</div>
            <div style="font-size: 0.85rem; color: #64748B; line-height: 1.8; margin-bottom: 1.5rem;">
                <b>1. Choose a Method:</b> Pick how you want the claim checked.<br>
                <b>2. Tell Us About the Claim:</b> Enter simple details like amount and dates.<br>
                <b>3. Get Your Result:</b> See whether the claim appears normal.<br>
                <b>4. Save & Download:</b> Keep a record for your files.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("Start Checking a Claim →", use_container_width=True, type="primary"):
            st.session_state["active_tab"] = "Assess"
            st.session_state["current_step"] = 1
            st.rerun()

# ==============================================================================
# TAB 2: RUN ASSESSMENT (STEP FLOW)
# ==============================================================================
elif st.session_state["active_tab"] == "Assess":
    curr_s = st.session_state["current_step"]
    st.markdown(f"""
    <div class="step-indicator-wrap">
        <div class="step-item {'step-completed' if curr_s > 1 else ('step-active' if curr_s == 1 else '')}">
            <div class="step-num">{'✓' if curr_s > 1 else '01'}</div>
            <span>01 Choose a Method</span>
        </div>
        <div class="step-arrow">&rarr;</div>
        <div class="step-item {'step-completed' if curr_s > 2 else ('step-active' if curr_s == 2 else '')}">
            <div class="step-num">{'✓' if curr_s > 2 else '02'}</div>
            <span>02 Tell Us About the Claim</span>
        </div>
        <div class="step-arrow">&rarr;</div>
        <div class="step-item {'step-completed' if curr_s > 3 else ('step-active' if curr_s == 3 else '')}">
            <div class="step-num">{'✓' if curr_s > 3 else '03'}</div>
            <span>03 Check the Information</span>
        </div>
        <div class="step-arrow">&rarr;</div>
        <div class="step-item {'step-completed' if st.session_state['is_saved'] else ('step-active' if curr_s == 4 else '')}">
            <div class="step-num">{'✓' if st.session_state['is_saved'] else '04'}</div>
            <span>04 Get Your Result</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # STEP 1: CHOOSE A METHOD
    # --------------------------------------------------------------------------
    if curr_s == 1:
        st.markdown("### 01 &mdash; Choose How You'd Like Us to Check the Claim")
        st.caption("Click any option below to choose the method you want to use.")
        
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        r2_c1, r2_c2 = st.columns(2)
        cols_all = [r1_c1, r1_c2, r1_c3, r2_c1, r2_c2]
        
        for idx, (k, algo) in enumerate(ALGORITHMS_INFO.items()):
            col = cols_all[idx % len(cols_all)]
            is_sel = (st.session_state["selected_algo_id"] == algo["id"])
            with col:
                st.markdown(f"""
                <div class="algo-grid-card {'algo-card-active' if is_sel else ''}">
                    <div>
                        <div class="algo-card-header">
                            <div class="algo-icon-box">{algo['icon']}</div>
                        </div>
                        <div class="algo-name">{algo['name']}</div>
                        <div class="algo-desc">{algo['friendly_description']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                btn_label = f"✓ Selected: {algo['name']}" if is_sel else f"Use {algo['name']} →"
                if st.button(btn_label, key=f"btn_algo_{algo['id']}", use_container_width=True, type="primary" if is_sel else "secondary"):
                    st.session_state["selected_algo_id"] = algo["id"]
                    st.session_state["current_step"] = 2
                    st.rerun()

    # --------------------------------------------------------------------------
    # STEP 2 & 3: ENTER DETAILS & CHECK RESULT
    # --------------------------------------------------------------------------
    elif curr_s in [2, 3, 4]:
        sel_algo = ALGORITHMS_INFO.get(st.session_state["selected_algo_id"], ALGORITHMS_INFO["random_forest"])

        # Active Method Header
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1.5px solid #E0E7FF; border-radius: 12px; padding: 0.85rem 1.35rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.5rem;">{sel_algo['icon']}</span>
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A;">Chosen Method: <span style="color: #4F46E5;">{sel_algo['name']}</span></div>
                    <div style="font-size: 0.8rem; color: #64748B;">{sel_algo['friendly_description']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_switch, _ = st.columns([1.5, 4])
        with col_switch:
            if st.button("← Choose a Different Method", use_container_width=True):
                st.session_state["current_step"] = 1
                st.rerun()

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

        col_form, col_result = st.columns([1.1, 0.9], gap="large")

        with col_form:
            st.markdown("### 📋 Tell Us About the Claim")
            st.caption("Enter the information as you would on a standard claim form.")

            with st.form("single_app_claim_form", clear_on_submit=False):
                # Driver Information
                with st.expander("👤 Driver Information", expanded=True):
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        age_years = st.number_input("How old is the driver?", min_value=18, max_value=85, value=35, step=1, help="Enter driver's age in years")
                        income_mode = st.selectbox(
                            "Driver Employment & Income Status",
                            options=["Employed / Earned Income", "Unemployed / No Reported Income (₹0)", "Student / Dependent (₹0)"],
                            help="Choose 'Unemployed / No Reported Income' if income is 0 or unverified."
                        )
                        if "No Reported Income" in income_mode or "Dependent" in income_mode:
                            annual_income_val = 0
                            st.caption("ℹ️ Annual income set to ₹0.")
                        else:
                            annual_income_val = st.number_input("Driver's Approximate Annual Income (₹)", min_value=0, max_value=5000000, value=45000, step=5000, help="Estimated annual income in Rupees")
                        gender_choice = st.selectbox("Driver's Gender", options=["Male", "Female"])
                        marital_choice = st.selectbox("Marital Status", options=["Married", "Single"])
                    with d_col2:
                        high_edu_choice = st.selectbox("Completed College or Higher Education?", options=["Yes", "No"])
                        past_claims_choice = st.selectbox("How many past claims has the driver made?", options=["None", "1 Past Claim", "2 Past Claims", "3 or more Claims"])
                        addr_change_choice = st.selectbox("Has the driver recently moved or changed address?", options=["No", "Yes"])
                        property_choice = st.selectbox("Homeowner or Property Status", options=["Owns Home / Property", "Renting"])

                # Vehicle Information
                with st.expander("🚗 Vehicle Details", expanded=True):
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        vehicle_price_val = st.number_input("Estimated Vehicle Value (₹)", min_value=10000, max_value=10000000, value=55000, step=5000, help="Approximate market price of the vehicle")
                        vehicle_age_years = st.number_input("How old is the vehicle? (Years)", min_value=0, max_value=25, value=4, step=1)
                    with v_col2:
                        category_choice = st.selectbox("Vehicle Body Type", options=["Sedan", "Hatchback / Compact", "SUV / Luxury"])
                        vehicle_color_choice = st.selectbox("Vehicle Color", options=["White", "Black", "Silver", "Blue", "Red", "Grey", "Other"])

                # Claim & Incident Details
                with st.expander("📄 Claim & Incident Details", expanded=True):
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        total_claim_val = st.number_input("Total Claim Amount (₹)", min_value=1000, max_value=5000000, value=48000, step=1000, help="Enter the total amount being claimed")
                        injury_claim_val = st.number_input("Injury Claim Amount (₹)", min_value=0, max_value=2000000, value=20000, step=1000, help="Enter the portion related to medical or injury expenses")
                        days_open_val = st.number_input("How many days has the claim been open?", min_value=1, max_value=365, value=45, step=1, help="Enter the number of days since the claim was opened")
                        police_report_choice = st.selectbox("Was an official police report filed?", options=["Yes", "No"])
                    with c_col2:
                        safety_stars = st.selectbox("Vehicle Safety Rating", options=["⭐⭐⭐⭐⭐ (5 Stars)", "⭐⭐⭐⭐☆ (4 Stars)", "⭐⭐⭐☆☆ (3 Stars)", "⭐⭐☆☆☆ (2 Stars)", "⭐☆☆☆☆ (1 Star)"], index=1)
                        witness_choice = st.selectbox("Was anyone there to witness the incident?", options=["Yes", "No"])
                        deductible_choice = st.selectbox("Policy Deductible Amount", options=["Standard (₹5,000)", "Medium (₹10,000)", "High (₹20,000)"])
                        defects_val = st.number_input("How many problems or irregularities were found in the form?", min_value=0, max_value=8, value=3, step=1)

                st.markdown("<div style='margin-top: 0.85rem;'></div>", unsafe_allow_html=True)
                run_btn = st.form_submit_button("✨ Check the Claim", use_container_width=True, type="primary")

            if run_btn:
                star_map = {
                    "⭐⭐⭐⭐⭐ (5 Stars)": 5,
                    "⭐⭐⭐⭐☆ (4 Stars)": 4,
                    "⭐⭐⭐☆☆ (3 Stars)": 3,
                    "⭐⭐☆☆☆ (2 Stars)": 2,
                    "⭐☆☆☆☆ (1 Star)": 1
                }
                deduct_map = {
                    "Standard (₹5,000)": 0.0,
                    "Medium (₹10,000)": 0.4,
                    "High (₹20,000)": 1.0
                }
                claims_map = {
                    "None": 0.0,
                    "1 Past Claim": 0.4,
                    "2 Past Claims": 0.8,
                    "3 or more Claims": 1.0
                }
                category_map = {"Sedan": 1, "Hatchback / Compact": 0, "SUV / Luxury": 2}
                color_map = {"White": 0, "Black": 1, "Silver": 2, "Blue": 3, "Red": 4, "Grey": 5, "Other": 6}

                model_inputs = {
                    'age_of_driver': age_years,
                    'annual_income': annual_income_val,
                    'gender': 1 if gender_choice == "Male" else 0,
                    'marital_status': 1 if marital_choice == "Married" else 0,
                    'high_education': 1 if high_edu_choice == "Yes" else 0,
                    'address_change': 1 if addr_change_choice == "Yes" else 0,
                    'property_status': 1 if property_choice == "Owns Home / Property" else 0,
                    'past_num_of_claims': claims_map.get(past_claims_choice, 0.0),
                    'past_claims': claims_map.get(past_claims_choice, 0.0),
                    'past_claims_label': past_claims_choice,
                    'vehicle_price': vehicle_price_val,
                    'age_of_vehicle': min(1.0, vehicle_age_years / 20.0),
                    'vehicle_category': category_map.get(category_choice, 1),
                    'vehicle_color': color_map.get(vehicle_color_choice, 3),
                    'total_claim': total_claim_val,
                    'injury_claim': injury_claim_val,
                    'days open': days_open_val,
                    'days_open': days_open_val,
                    'police_report': 1 if police_report_choice == "Yes" else 0,
                    'safety_rating': star_map.get(safety_stars, 4),
                    'safety_stars_label': safety_stars.split(" ")[0],
                    'witness_present': 1 if witness_choice == "Yes" else 0,
                    'policy deductible': deduct_map.get(deductible_choice, 0.4),
                    'policy_deductible': deduct_map.get(deductible_choice, 0.4),
                    'form defects': defects_val,
                    'form_defects': defects_val
                }

                readable_summary = {
                    "Driver Age": f"{age_years} years old",
                    "Annual Income": "₹ 0 (Unemployed / Unreported)" if annual_income_val == 0 else f"₹ {annual_income_val:,}",
                    "Total Claim Amount": f"₹ {total_claim_val:,}",
                    "Injury Claim Amount": f"₹ {injury_claim_val:,}",
                    "Days Claim Open": f"{days_open_val} days",
                    "Police Report": police_report_choice,
                    "Witness Present": witness_choice,
                    "Vehicle Rating": safety_stars.split(" ")[0],
                    "Past Claims": past_claims_choice
                }

                with st.spinner("Checking your claim..."):
                    res = execute_claim_check(sel_algo["id"], model_inputs)

                st.session_state["last_result"] = res
                st.session_state["last_human_inputs"] = readable_summary
                st.session_state["raw_inputs"] = model_inputs
                st.session_state["current_step"] = 3
                st.session_state["is_saved"] = False
                st.rerun()

        # Output Display Column
        with col_result:
            st.markdown("### 🎯 Your Result")
            st.caption("Here is what the system found based on the information provided.")

            last_res = st.session_state.get("last_result")

            if last_res:
                is_fraud = last_res.get("is_fraud", False)
                prob_fraud = float(last_res.get("probability_fraud", last_res.get("probability", 25.0)))
                prob_normal = float(last_res.get("probability_normal", 100.0 - prob_fraud))

                # Friendly Status Banner (Section 23)
                if is_fraud:
                    st.markdown(f"""
                    <div class="result-banner-review">
                        <div class="result-title">⚠️ {last_res['friendly_title']}</div>
                        <div class="result-subtitle">{last_res['friendly_explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-banner-safe">
                        <div class="result-title">🟢 {last_res['friendly_title']}</div>
                        <div class="result-subtitle">{last_res['friendly_explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Gauge Meter - Estimated Fraud Probability (Section 5 & 24)
                meter_color = "#DC2626" if is_fraud else "#10B981"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_fraud,
                    number={'suffix': "%", 'font': {'size': 34, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
                    title={'text': "Estimated Fraud Probability", 'font': {'size': 13, 'color': '#64748B', 'family': 'Plus Jakarta Sans'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                        'bar': {'color': meter_color, 'thickness': 0.26},
                        'bgcolor': "white",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.12)'},
                            {'range': [35, 50], 'color': 'rgba(245, 158, 11, 0.12)'},
                            {'range': [50, 100], 'color': 'rgba(220, 38, 38, 0.12)'}
                        ],
                        'threshold': {'line': {'color': "#DC2626", 'width': 3}, 'thickness': 0.8, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=25, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Probability Breakdown Metrics (Section 5 & 24)
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.metric(label="Estimated Fraud Probability", value=f"{prob_fraud:.1f}%")
                with p_c2:
                    st.metric(label="Estimated Normal Probability", value=f"{prob_normal:.1f}%")
                st.caption("ℹ️ Standard decision threshold: **50.0%**. Claims exceeding 50.0% fraud probability are flagged for review.")

                # Information You Entered
                st.markdown("#### 📋 Information You Entered")
                human_inputs_shown = st.session_state.get("last_human_inputs", {})
                if human_inputs_shown:
                    with st.container(border=True):
                        for k, v in human_inputs_shown.items():
                            c_lbl, c_val = st.columns([1.4, 1])
                            c_lbl.write(f"**{k}**")
                            c_val.write(f"{v}")
                        c_mlbl, c_mval = st.columns([1.4, 1])
                        c_mlbl.write("**Method Used**")
                        c_mval.write(f"**{last_res['algorithm']}**")

                # Factual Information Considered (Section 25)
                st.markdown("#### 🔍 Key Information Considered by Model")
                factors = last_res.get("considered_factors", [])
                with st.container(border=True):
                    for f in factors:
                        fc1, fc2 = st.columns([3, 1.4])
                        with fc1:
                            st.markdown(f"**{f.get('factor')}**")
                            st.caption(f.get('note'))
                        with fc2:
                            st.markdown(f"`{f.get('value')}`")

                # Actions
                st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
                act_c1, act_c2 = st.columns(2)

                with act_c1:
                    if not st.session_state["is_saved"]:
                        if st.button("💾 Save Result", use_container_width=True, type="primary"):
                            save_rec = {
                                "algorithm": last_res["algorithm"],
                                "inputs": st.session_state.get("last_human_inputs", {}),
                                "prediction": "May Need Review" if is_fraud else "Appears Normal",
                                "friendly_title": last_res["friendly_title"],
                                "is_fraud": is_fraud,
                                "probability": prob_fraud
                            }
                            if save_history_record(save_rec):
                                st.session_state["is_saved"] = True
                                st.session_state["current_step"] = 4
                                st.toast("Your result has been saved successfully.", icon="💾")
                                st.rerun()
                    else:
                        st.button("✓ Result Saved", disabled=True, use_container_width=True)

                with act_c2:
                    if st.button("🔄 Check Another Claim", use_container_width=True):
                        st.session_state["last_result"] = None
                        st.session_state["last_human_inputs"] = None
                        st.session_state["current_step"] = 1
                        st.session_state["is_saved"] = False
                        st.rerun()

                # Developer Technical Details (Section 22)
                with st.expander("⚙️ Technical Details (Developer Debug Panel)", expanded=False):
                    diag = last_res.get("diagnostics", {})
                    st.markdown("##### 🛠️ ML Inference Diagnostics")
                    st.write(f"**Selected Algorithm:** `{last_res['algorithm']}`")
                    st.write(f"**Model File:** `{diag.get('model_file', 'model.pkl')}`")
                    st.write(f"**Model Class:** `{diag.get('model_class', 'Estimator')}`")
                    st.write(f"**Target Classes (`classes_`):** `{diag.get('model_classes', [0, 1])}` *(0 = Legitimate, 1 = Fraud)*")
                    st.write(f"**Raw Model Prediction:** `{diag.get('raw_prediction')} ({'Class 1: Fraud' if diag.get('raw_prediction') == 1 else 'Class 0: Legitimate'})`")
                    st.write(f"**Decision Threshold:** `{diag.get('decision_threshold', '50.0%')}`")
                    st.write(f"**Class Probabilities:**")
                    st.json(diag.get("class_probabilities", {}))
                    st.write(f"**Input Shape:** `{diag.get('input_shape')}`")
                    st.write(f"**StandardScaler Applied:** `{diag.get('requires_scaler', False)}`")
                    st.write(f"**Task-5 Accuracy:** `{diag.get('accuracy', '78%')}`")
                    st.write("**Processed Normalized Vector (29 Features):**")
                    st.json(diag.get("internal_features", {}))
                    
                    export_data = {
                        "method_used": last_res["algorithm"],
                        "result": "May Need Review" if is_fraud else "Appears Normal",
                        "explanation": last_res["friendly_explanation"],
                        "fraud_probability": prob_fraud,
                        "normal_probability": prob_normal,
                        "date_checked": last_res["timestamp"],
                        "details_entered": st.session_state.get("last_human_inputs", {})
                    }
                    st.download_button(
                        label="📥 Download Result Summary (JSON)",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"claim_check_{int(datetime.now().timestamp())}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            else:
                st.markdown("""
                <div style="background: #FFFFFF; border: 2px dashed #E2E8F0; border-radius: 16px; padding: 4rem 2rem; text-align: center; color: #64748B;">
                    <div style="font-size: 2.75rem; margin-bottom: 0.75rem;">🛡️</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1E293B; margin-bottom: 0.35rem;">Waiting for Claim Details</div>
                    <div style="font-size: 0.85rem; color: #64748B; max-width: 320px; margin: 0 auto 1.5rem auto;">
                        Enter a few details about the claim on the left and click <b>'Check the Claim'</b> to get your result.
                    </div>
                    <div style="display: flex; justify-content: center; gap: 1rem; font-size: 0.78rem; font-weight: 600; color: #94A3B8;">
                        <span>✓ Fast & Easy</span>
                        <span>✓ Clear Explanations</span>
                        <span>✓ Safe & Secure</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: PREDICTION HISTORY
# ==============================================================================
elif st.session_state["active_tab"] == "History":
    st.markdown("### 📜 My Previous Checks")
    st.caption("Review past claim checks, inspect details you entered, and download summaries.")

    history_records = read_history_records()

    total_preds = len(history_records)
    review_preds = sum(1 for r in history_records if r.get("is_fraud", False) or "review" in str(r.get("prediction", "")).lower())
    normal_preds = total_preds - review_preds

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Total Claims Checked</span>
            <span class="kpi-val">{total_preds}</span>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Claims Cleared (Normal)</span>
            <span class="kpi-val" style="color: #059669;">{normal_preds}</span>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Claims Needing Review</span>
            <span class="kpi-val" style="color: #DC2626;">{review_preds}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    if not history_records:
        st.markdown("""
        <div style="background: #FFFFFF; border: 2px dashed #E2E8F0; border-radius: 16px; padding: 4rem 2rem; text-align: center; color: #64748B;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">📂</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">No Saved Checks Yet</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.25rem;">
                Check a claim in the <b>Check a Claim</b> tab and click <b>Save Result</b> to store it here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        act_c1, act_c2, _ = st.columns([1.5, 1.5, 3])
        with act_c1:
            if st.button("🗑️ Clear All Saved Results", use_container_width=True):
                if clear_all_records():
                    st.toast("Your saved results have been removed.", icon="🗑️")
                    st.rerun()
        with act_c2:
            csv_rows = []
            for r in history_records:
                row_flat = {
                    "Record_ID": r.get("id"),
                    "Method_Used": r.get("algorithm"),
                    "Result": r.get("prediction"),
                    "Date": r.get("timestamp", "")
                }
                if isinstance(r.get("inputs"), dict):
                    for k, v in r["inputs"].items():
                        row_flat[k] = v
                csv_rows.append(row_flat)
            df_hist = pd.DataFrame(csv_rows)
            csv_bytes = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export as Spreadsheet (CSV)",
                data=csv_bytes,
                file_name=f"claim_check_history_{int(datetime.now().timestamp())}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        for rec in history_records:
            is_f = rec.get("is_fraud", False) or "review" in str(rec.get("prediction", "")).lower()
            rec_id = rec.get("id", "N/A")
            badge_color = "#FEF2F2" if is_f else "#ECFDF5"
            badge_text_color = "#DC2626" if is_f else "#059669"
            border_col = "#FCA5A5" if is_f else "#A7F3D0"
            result_str = "May Need Review" if is_f else "Appears Normal"

            with st.container(border=True):
                h_c1, h_c2 = st.columns([4, 1])
                with h_c1:
                    icon_stat = '🔴' if is_f else '🟢'
                    st.markdown(f"#### {icon_stat} {rec.get('algorithm', 'Method')} &mdash; `{result_str}`")
                    st.caption(f"Date Checked: {rec.get('timestamp', '')} &bull; ID: `{rec_id}`")
                with h_c2:
                    if st.button("Delete", key=f"del_{rec_id}", use_container_width=True):
                        if delete_history_record(rec_id):
                            st.toast("Your saved result has been removed.", icon="🗑️")
                            st.rerun()

                with st.expander(f"🔍 View Details for this Claim"):
                    inputs_rec = rec.get("inputs", {})
                    if isinstance(inputs_rec, dict) and inputs_rec:
                        for k, v in inputs_rec.items():
                            st.write(f"• **{k}:** {v}")
                    else:
                        st.write("No detailed notes logged for this entry.")
