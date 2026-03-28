import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
)

# ── Load model & vectorizer (cached) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 720px; margin: auto; }
    .result-spam {
        background: #8C392B;
        border-left: 5px solid #e53935;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.1rem;
        margin-top: 12px;
    }
    .result-ham {
        background: #1b5e20;
        border-left: 5px solid #43a047;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.1rem;
        margin-top: 12px;
    }
    .confidence-label { font-size: 0.85rem; color: #555; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📱 SMS Spam Detector")
st.caption("Powered by Multinomial Naive Bayes + TF-IDF · Paste any SMS message below")

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
message = st.text_area(
    "Enter SMS message",
    placeholder="e.g.  Congratulations! You've won a £1000 prize. Call now to claim!",
    height=130,
    label_visibility="collapsed",
)

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚨 Spam example"):
        st.session_state["example"] = "WINNER!! You have been selected for a cash prize of $5000. Call 08712300150 NOW to claim!"
with col2:
    if st.button("✅ Ham example"):
        st.session_state["example"] = "Hey, are you coming to the meeting at 3pm today? Let me know if you need a ride."
with col3:
    if st.button("🔁 Another spam"):
        st.session_state["example"] = "Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"

# Fill textarea if example was clicked
if "example" in st.session_state and not message:
    message = st.session_state.pop("example")
    st.rerun()

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Analyse Message", type="primary", use_container_width=True)

if predict_btn:
    if not message.strip():
        st.warning("⚠️ Please enter a message to analyse.")
    else:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]  # [ham_prob, spam_prob]

        spam_prob = proba[1] * 100
        ham_prob  = proba[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-spam">
                🚨 <strong>SPAM</strong> — This message looks like spam!
            </div>
            <div class="confidence-label">Confidence: Spam {spam_prob:.1f}% &nbsp;|&nbsp; Ham {ham_prob:.1f}%</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                ✅ <strong>HAM</strong> — This message looks legitimate.
            </div>
            <div class="confidence-label">Confidence: Ham {ham_prob:.1f}% &nbsp;|&nbsp; Spam {spam_prob:.1f}%</div>
            """, unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("&nbsp;")
        col_h, col_s = st.columns(2)
        col_h.metric("Ham probability",  f"{ham_prob:.1f}%")
        col_s.metric("Spam probability", f"{spam_prob:.1f}%")

        st.progress(int(spam_prob), text=f"Spam likelihood: {spam_prob:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: MultinomialNB · Vectorizer: TF-IDF · scikit-learn")