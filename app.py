# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"
META_PATH = "meta.json"

TITLE = "Infective Endocarditis Cerebral Embolism Risk Calculator"

st.set_page_config(page_title=TITLE, layout="wide")

# ---------- style ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1100px; }
    h1 { margin-bottom: 0.25rem; }
    .subtle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.2rem; }
    .card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }
    .result {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.2rem 0 0.4rem 0;
    }
    .caption { color: #6b7280; font-size: 0.92rem; }
    div.stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
meta = load_meta()

st.title(TITLE)
st.markdown(
    "<div class='subtle'>Enter patient values to estimate the probability of cerebral embolism.</div>",
    unsafe_allow_html=True
)

feature_order = meta["feature_order"]
fields = meta["fields"]

def render_field(col_name: str):
    cfg = fields[col_name]
    label = cfg.get("label", col_name)

    if cfg["type"] == "binary":
        default = int(cfg.get("default", 0))
        options = ["No", "Yes"]
        default_idx = 1 if default == 1 else 0
        val = st.selectbox(label, options=options, index=default_idx)
        return 1 if val == "Yes" else 0

    # numeric
    vmin = float(cfg.get("min", 0.0))
    vmax = float(cfg.get("max", 1.0))
    vdef = float(cfg.get("default", (vmin + vmax) / 2))
    step = float(cfg.get("step", 0.1))

    # clamp default into range
    vdef = min(max(vdef, vmin), vmax)

    # age looks better as int
    if col_name.lower() == "age":
        return st.number_input(label, min_value=int(np.floor(vmin)), max_value=int(np.ceil(vmax)),
                               value=int(round(vdef)), step=1)
    else:
        return st.number_input(label, min_value=vmin, max_value=vmax, value=vdef, step=step, format="%.6f")

# ---------- layout ----------
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("risk_form", clear_on_submit=False):
        c1, c2 = st.columns(2, gap="medium")
        values = {}
        for i, feat in enumerate(feature_order):
            with (c1 if i % 2 == 0 else c2):
                values[feat] = render_field(feat)

        submitted = st.form_submit_button("Estimate risk")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Estimated risk**", unsafe_allow_html=True)
    placeholder = st.empty()
    st.markdown("<div class='caption'>Probability output from the trained model.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- inference ----------
if submitted:
    X = pd.DataFrame([[values[f] for f in feature_order]], columns=feature_order)

    # Make sure numeric dtypes are sane
    for f in feature_order:
        if fields[f]["type"] == "numeric":
            X[f] = pd.to_numeric(X[f], errors="coerce")

    try:
        proba = float(model.predict_proba(X)[0, 1])
        proba = min(max(proba, 0.0), 1.0)
        placeholder.markdown(
            f"<div class='result'>{proba*100:.1f}%</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown(
    "<div class='caption'>For research/educational use only. Not medical advice.</div>",
    unsafe_allow_html=True
)
