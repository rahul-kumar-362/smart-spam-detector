import streamlit as st
from models.model import predict_mail

# ---------------- SESSION STATE ----------------
if "count" not in st.session_state:
    st.session_state.count = 0

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#4CAF50;
}
.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}
.result-box {
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}
.spam {
    background-color:#ff4b4b33;
    color:#ff4b4b;
}
.genuine {
    background-color:#4CAF5033;
    color:#4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">📧 Smart Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI powered message classification system</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
msg = st.text_area("Enter your text:", height=150)

# ---------------- ANALYZE BUTTON ----------------
if st.button("Analyze Message"):

    # Validation
    if msg.strip() == "":
        st.warning("⚠ Please enter some text first.")
        st.stop()

    if len(msg) > 500:
        st.error("Text too long. Limit is 500 characters.")
        st.stop()

    # Prediction
    with st.spinner("Analyzing message..."):
        pred, prob = predict_mail(msg)

    # Increase counter
    st.session_state.count += 1

    # Label
    label = "Spam 🚨" if pred == "spam" else "Genuine ✅"
    css_class = "spam" if pred == "spam" else "genuine"

    # Result box
    st.markdown(
        f'<div class="result-box {css_class}">{label}</div>',
        unsafe_allow_html=True
    )

    # Confidence
    st.write("Confidence Score:")
    st.progress(float(prob))
    st.write(f"**{round(prob*100,2)}% confidence**")

    # Confidence Interpretation
    if prob > 0.85:
        st.success("Very confident prediction")
    elif prob > 0.60:
        st.info("Moderately confident")
    else:
        st.warning("Low confidence prediction")

    # ---------------- EXPLAINABLE AI ----------------
    spam_words = [
        "free","win","winner","prize","claim","click","offer","urgent","money","reward",
        "lottery","bonus","cash","deal","discount","limited","exclusive","act now","buy now",
        "guaranteed","gift","congratulations","earn","income","profit","cheap","credit","loan",
        "approved","extra","amazing","vip","secret","priority","alert","final notice",
        "last chance","hurry","expire","deadline","today only","instant","fast","quick",
        "limited time","best price","jackpot","million","earn money","work from home",
        "financial freedom","investment","crypto","bitcoin","forex","verify account",
        "confirm details","update account","account suspended","login now","reset password",
        "click link","download now","install","limited stock","free trial","free access",
        "free membership","instant approval","selected user","exclusive invitation"
    ]

    found = [word for word in spam_words if word in msg.lower()]

    if found:
        st.write("⚠ Spam trigger words detected:")
        st.code(", ".join(found))

# ---------------- CLEAR BUTTON ----------------
if st.button("Clear"):
    st.rerun()

# ---------------- DASHBOARD ----------------
st.divider()
st.subheader("System Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Predictions", st.session_state.count)

with col2:
    st.metric("Model Accuracy", "96.68%")

# ---------------- MODEL METRICS ----------------
st.subheader("Model Performance Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Precision", "97%")
col2.metric("Recall", "96%")
col3.metric("F1 Score", "96.5%")
