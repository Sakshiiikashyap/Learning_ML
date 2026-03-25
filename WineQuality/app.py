import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0608 0%, #1a0a10 40%, #0d0608 100%);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(180,30,60,0.2);
}
.wine-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem; font-weight: 600;
    color: #e8c4b8; letter-spacing: 0.02em;
    margin-bottom: 0; line-height: 1.1;
}
.wine-subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic; font-size: 1.1rem;
    color: rgba(180,90,90,0.8);
    margin-top: 0.2rem; margin-bottom: 2rem;
}
.stButton > button {
    background: linear-gradient(135deg, #8b1a2e 0%, #c0392b 100%);
    color: #f5e6e0; font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 1rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    border: none; border-radius: 8px;
    padding: 0.75rem 2.5rem; width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(139,26,46,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #a0213a 0%, #d44000 100%);
    box-shadow: 0 6px 28px rgba(139,26,46,0.6);
    transform: translateY(-1px);
}
.section-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem; color: #c98b8b;
    border-bottom: 1px solid rgba(180,30,60,0.3);
    padding-bottom: 0.4rem; margin-bottom: 1rem;
    letter-spacing: 0.04em;
}
.result-box {
    background: linear-gradient(135deg, rgba(139,26,46,0.15) 0%, rgba(90,15,25,0.25) 100%);
    border: 1px solid rgba(180,30,60,0.5);
    border-radius: 16px; padding: 2rem;
    text-align: center; backdrop-filter: blur(12px);
}
.result-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem; font-weight: 600;
    color: #e8c4b8; line-height: 1;
}
.result-label {
    font-size: 0.95rem; color: rgba(200,150,140,0.8);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-top: 0.3rem;
}
label { color: #c9a89a !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts from same folder as app.py ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def load_pkl(filename):
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Could not load {filename}: {e}")
                return None

    model         = load_pkl("model.pkl")
    scaler        = load_pkl("scaler.pkl")
    label_encoder = load_pkl("label_encoder.pkl")
    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()

# ── Validate what was loaded ───────────────────────────────────────────────────
# scaler must have a .transform() method — if it's a list, ignore it
if scaler is not None and not hasattr(scaler, "transform"):
    scaler = None   # corrupted / wrong object saved

# label_encoder must have .inverse_transform() — otherwise ignore
if label_encoder is not None and not hasattr(label_encoder, "inverse_transform"):
    label_encoder = None


# ── Class label mapping (from your notebook) ──────────────────────────────────
# LabelEncoder encodes alphabetically: high=0, low=1, medium=2
# (your simplify_quality maps: ≤4 → low, 5-6 → medium, 7+ → high)
LABEL_MAP = {0: "High 🌟", 1: "Low ⚠️", 2: "Medium ✓"}
LABEL_COLOR = {0: "#5cb85c", 1: "#d9534f", 2: "#f0ad4e"}
LABEL_EMOJI = {0: "🌟", 1: "⚠️", 2: "✓"}



# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(features: np.ndarray):
    X = features.copy().astype(float)

    # Apply scaler if available and valid
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"Scaler failed, using raw features: {e}")

    raw = model.predict(X)[0]

    # Decode with label_encoder if available
    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([int(raw)])[0]
            return int(raw), decoded
        except Exception:
            pass

    # Fallback: use built-in map
    decoded = LABEL_MAP.get(int(raw), f"Class {raw}")
    return int(raw), decoded


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<h1 class="wine-title">Wine Quality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="wine-subtitle">🍷 Red Wine</p>', unsafe_allow_html=True)

tab_predict, tab_about = st.tabs(
    ["🔬 Single Prediction", "ℹ️ About"]
)

# ════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<p class="section-label">Acidity & Sugars</p>', unsafe_allow_html=True)
        fixed_acidity    = st.slider("Fixed Acidity (g/L)",    4.0,  16.0,  7.4,  0.1)
        volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.1,   1.6,  0.52, 0.01)
        citric_acid      = st.slider("Citric Acid (g/L)",      0.0,   1.0,  0.26, 0.01)
        residual_sugar   = st.slider("Residual Sugar (g/L)",   0.5,  65.0,  2.2,  0.1)

    with c2:
        st.markdown('<p class="section-label">Minerals & Gases</p>', unsafe_allow_html=True)
        chlorides    = st.slider("Chlorides (g/L)",             0.01, 0.61, 0.079, 0.001, format="%.3f")
        free_sulfur  = st.slider("Free Sulfur Dioxide (mg/L)",  1.0,  72.0, 14.0,  1.0)
        total_sulfur = st.slider("Total Sulfur Dioxide (mg/L)", 6.0, 289.0, 46.0,  1.0)

    with c3:
        st.markdown('<p class="section-label">Physical Properties</p>', unsafe_allow_html=True)
        density   = st.slider("Density (g/mL)",  0.990, 1.004, 0.9978, 0.0001, format="%.4f")
        pH        = st.slider("pH",              2.7,   4.0,   3.31,   0.01)
        sulphates = st.slider("Sulphates (g/L)", 0.3,   2.0,   0.62,   0.01)
        alcohol   = st.slider("Alcohol (% vol)", 8.0,   15.0,  10.4,   0.1)

    st.markdown("")
    predict_col, result_col = st.columns(2)

    # NOTE: your notebook cast to int before training, so we do the same
    features = np.array([[
        int(fixed_acidity), int(volatile_acidity), int(citric_acid), int(residual_sugar),
        int(chlorides), int(free_sulfur), int(total_sulfur), int(density),
        int(pH), int(sulphates), int(alcohol)
    ]])

    feature_names = [
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
        "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
    ]

    with predict_col:
        predict_btn = st.button("🍷 Predict Wine Quality", use_container_width=True)

    if predict_btn:
        if model is None:
            st.warning("Model not found. Place `model.pkl` in the same folder as `app.py`.")
        else:
            with st.spinner("Analysing..."):
                try:
                    class_code, label = predict(features)
                    color = LABEL_COLOR.get(class_code, "#e8c4b8")

                    with result_col:
                        st.markdown(
                            f"""<div class="result-box">
                                <div class="result-score" style="color:{color}">{label}</div>
                                <div class="result-label">Predicted Quality Class</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    # Gauge — map class to 0-10 range for display
                    score_display = {0: 8, 1: 3, 2: 6}.get(class_code, 5)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score_display,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": f"Quality: {label}", "font": {"color": "#e8c4b8", "size": 14}},
                        number={"font": {"color": color, "size": 40}, "suffix": "/10"},
                        gauge={
                            "axis": {"range": [0, 10], "tickcolor": "#c9a89a"},
                            "bar": {"color": color},
                            "bgcolor": "rgba(0,0,0,0)",
                            "bordercolor": "rgba(180,30,60,0.3)",
                            "steps": [
                                {"range": [0, 4],  "color": "rgba(100,20,20,0.3)"},
                                {"range": [4, 7],  "color": "rgba(150,60,30,0.3)"},
                                {"range": [7, 10], "color": "rgba(100,180,80,0.2)"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#e8c4b8"}, height=280,
                        margin=dict(t=40, b=10, l=20, r=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Radar chart
                    st.markdown("### 🕸️ Feature Profile")
                    raw_vals = [
                        fixed_acidity, volatile_acidity, citric_acid,
                        residual_sugar, chlorides, free_sulfur,
                        total_sulfur, density, pH, sulphates, alcohol,
                    ]
                    norm_vals = [
                        fixed_acidity / 16, volatile_acidity / 1.6, citric_acid,
                        residual_sugar / 65, chlorides / 0.61, free_sulfur / 72,
                        total_sulfur / 289, (density - 0.990) / 0.014,
                        (pH - 2.7) / 1.3, sulphates / 2.0, alcohol / 15,
                    ]
                    radar = go.Figure(go.Scatterpolar(
                        r=norm_vals + [norm_vals[0]],
                        theta=feature_names + [feature_names[0]],
                        fill="toself",
                        fillcolor="rgba(139,26,46,0.2)",
                        line=dict(color=color, width=2),
                    ))
                    radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0, 1], color="#c9a89a",
                                            gridcolor="rgba(180,30,60,0.2)"),
                            angularaxis=dict(color="#c9a89a", gridcolor="rgba(180,30,60,0.2)"),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#e8c4b8"}, height=380,
                        margin=dict(t=30, b=30, l=60, r=60),
                        showlegend=False,
                    )
                    st.plotly_chart(radar, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.caption("Check that model.pkl is in the same folder and trained on the same features.")

# ════════════════════════════════════════════════════════════
# TAB 2 — About
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### About This App

This Streamlit app wraps your trained **Wine Quality Predictor** — a classifier trained on the
UCI Red Wine Quality dataset that predicts quality as **High**, **Medium**, or **Low**.

#### 🔄Training Pipeline (from notebook)

1. Quality score ≤ 4 → **Low**, 5–6 → **Medium**, 7+ → **High**
2. LabelEncoder encodes alphabetically: **High=0, Low=1, Medium=2**
3. Features cast to `int` before training
4. StandardScaler applied to training data
5. Model trained (RandomForest achieved best accuracy ~83%)

#### 📁 Required Files (same folder as `app.py`)

| File | Required | Purpose |
|---|---|---|
| `model.pkl` | ✅ Yes | Trained classifier |
| `scaler.pkl` | Optional | StandardScaler for features |
| `label_encoder.pkl` | Optional | LabelEncoder for class names |

#### 📊 Input Features (11 — cast to int internally)

| Feature | Description |
|---|---|
| Fixed Acidity | Non-volatile tartaric acid |
| Volatile Acidity | Acetic acid level |
| Citric Acid | Freshness |
| Residual Sugar | Sugar after fermentation |
| Chlorides | Salt content |
| Free SO₂ | Antimicrobial agent |
| Total SO₂ | Total sulfur dioxide |
| Density | Alcohol + sugar proxy |
| pH | Acidity (2.7–4.0) |
| Sulphates | Stabiliser |
| Alcohol | % alcohol by volume |
    """)