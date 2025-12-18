# app.py
# Streamlit app: Infective Endocarditis Cerebral Embolism Risk Calculator

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Infective Endocarditis Cerebral Embolism Risk Calculator"


# ----------------------------
# Page config + style
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 2.5rem; }
      .stForm { border: 1px solid rgba(49,51,63,0.2); border-radius: 18px; padding: 18px; }
      div[data-testid="stMetricValue"] { font-size: 44px; }
      .subtle { color: rgba(49,51,63,0.65); font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.markdown(
    '<div class="subtle">Enter patient values below and click <b>Calculate risk</b>. '
    'All numeric fields must be <b>greater than 0</b>.</div>',
    unsafe_allow_html=True,
)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.pkl"
META_PATH = HERE / "meta.json"


# ----------------------------
# Utilities
# ----------------------------
@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def parse_positive_float(raw: str, field_name: str):
    """
    - Allow blank (returns None)
    - Must be numeric
    - Must be > 0
    - Round to 1 decimal for computation
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        x = float(s)
    except ValueError:
        raise ValueError(f"{field_name}: please enter a valid number.")
    if not (x > 0):
        raise ValueError(f"{field_name}: must be greater than 0.")
    return float(np.round(x, 1))


def yes_no_select(label: str, key: str):
    """
    Dropdown with a blank default so the page is clean on load.
    Returns: None (not chosen), 0 (No), 1 (Yes)
    """
    choice = st.selectbox(label, ["", "No", "Yes"], index=0, key=key)
    if choice == "":
        return None
    return 1 if choice == "Yes" else 0


def infer_feature_order(model, meta: dict):
    """
    Try (in order):
      1) model.feature_names_in_ (sklearn)
      2) meta['feature_names'] / meta['feature_order']
      3) fallback hardcoded order
    """
    if hasattr(model, "feature_names_in_"):
        names = list(getattr(model, "feature_names_in_"))
        if names:
            return names

    for k in ["feature_names", "feature_order", "features"]:
        if k in meta:
            if isinstance(meta[k], list) and meta[k]:
                # meta["features"] might be list of dicts
                if isinstance(meta[k][0], dict) and "name" in meta[k][0]:
                    return [d["name"] for d in meta[k]]
                # meta["feature_names"] might be list of strings
                if isinstance(meta[k][0], str):
                    return meta[k]

    # Fallback: your current app fields
    return ["Age", "AST_ALT", "ALB", "LDH_U_L", "D_DI", "Past_embolism", "Staphylococcus_aureus"]


def proba_of_positive_class(model, X: pd.DataFrame) -> float:
    """
    Robustly pick P(y=1).
    """
    proba = model.predict_proba(X)
    # If model has classes_, choose class==1; else assume column 1 is positive
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return float(proba[0, idx])
    return float(proba[0, 1])


# ----------------------------
# Load model/meta
# ----------------------------
if not MODEL_PATH.exists():
    st.error("model.pkl not found in the app folder.")
    st.stop()

model = load_model(MODEL_PATH)
meta = load_meta(META_PATH)
feature_order = infer_feature_order(model, meta)


# ----------------------------
# Form UI
# ----------------------------
with st.form("risk_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        age_raw = st.text_input("Age (years)", value="", placeholder="e.g. 58.0", key="Age")
        alb_raw = st.text_input("Albumin (g/L)", value="", placeholder="e.g. 33.9", key="ALB")
        ddi_raw = st.text_input("D-dimer (mg/L)", value="", placeholder="e.g. 1.5", key="D_DI")
        staph = yes_no_select("Staphylococcus aureus", "Staphylococcus_aureus")

    with c2:
        astalt_raw = st.text_input("AST/ALT ratio", value="", placeholder="e.g. 1.1", key="AST_ALT")
        ldh_raw = st.text_input("LDH (U/L)", value="", placeholder="e.g. 254.0", key="LDH_U_L")
        past = yes_no_select("Past embolism", "Past_embolism")

    submitted = st.form_submit_button("Calculate risk")


# ----------------------------
# Prediction
# ----------------------------
if submitted:
    try:
        age = parse_positive_float(age_raw, "Age")
        astalt = parse_positive_float(astalt_raw, "AST/ALT ratio")
        alb = parse_positive_float(alb_raw, "Albumin")
        ldh = parse_positive_float(ldh_raw, "LDH")
        ddi = parse_positive_float(ddi_raw, "D-dimer")

        # Required checks (keep it strict and clear)
        missing = []
        if age is None: missing.append("Age")
        if astalt is None: missing.append("AST/ALT ratio")
        if alb is None: missing.append("Albumin")
        if ldh is None: missing.append("LDH")
        if ddi is None: missing.append("D-dimer")
        if past is None: missing.append("Past embolism (Yes/No)")
        if staph is None: missing.append("Staphylococcus aureus (Yes/No)")

        if missing:
            st.error("Please fill in: " + ", ".join(missing))
            st.stop()

        row = {
            "Age": age,
            "AST_ALT": astalt,
            "ALB": alb,
            "LDH_U_L": ldh,
            "D_DI": ddi,
            "Past_embolism": past,
            "Staphylococcus_aureus": staph,
        }

        # Build X with the exact feature order expected by the model
        X = pd.DataFrame([{name: row.get(name, np.nan) for name in feature_order}])

        # Safety: if any NaN remains, fail loudly (means feature name mismatch)
        if X.isna().any().any():
            bad_cols = X.columns[X.isna().any()].tolist()
            st.error(
                "Feature mismatch between UI and model. Missing columns: "
                + ", ".join(bad_cols)
                + ".\n\nCheck meta.json / model feature names."
            )
            st.stop()

        risk = proba_of_positive_class(model, X)

        st.markdown("---")
        st.metric("Estimated risk (probability)", f"{risk*100:.1f}%")

        st.caption("Note: This calculator is for research/educational use only and does not replace clinical judgment.")

    except Exception as e:
        st.error(str(e))
