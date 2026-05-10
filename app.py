# ─────────────────────────────────────────────────────────────────────────────
# app.py  —  Suicidal Tendency Prediction GUI
# Run:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────
import nltk
import os

# Download NLTK data for Streamlit Cloud
nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Text Screener",
    page_icon="🧠",
    layout="wide"
)
STOP_WORDS = set(stopwords.words("english"))
NEGATIONS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "noone", "none", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't", "without"
}


def handle_negations(tokens):
    result, negate = [], False
    for token in tokens:
        if token in NEGATIONS:
            negate = True
            result.append(token)
        elif negate:
            result.append(token + "_NEG")
            if token in {".", ",", "!", "?", ";"}:
                negate = False
        else:
            result.append(token)
    return result


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = handle_negations(tokens)
    tokens = [t for t in tokens if t not in STOP_WORDS or t.endswith("_NEG")]
    return " ".join(tokens)


@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("lr_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    return tfidf, model, threshold


# ── Mental Health Resources (ALWAYS shown) ────────────────────────────────────
def show_resources():
    st.markdown("""
    <div style='background:#1a472a;padding:16px;border-radius:10px;border-left:6px solid #2ecc71;'>
    <h3 style='color:#2ecc71;margin:0;'>💚 Mental Health Resources</h3>
    <p style='color:#ecf0f1;margin-top:8px;'>
    These resources are always available to you, regardless of this tool's output:
    </p>
    <ul style='color:#ecf0f1;'>
    <li>🇧🇩 <b>Kaan Pete Roi (Bangladesh):</b> +8801779-554391</li>
    <li>🌐 <b>Crisis Text Line:</b> Text HOME to 741741</li>
    <li>📞 <b>International hotlines:</b> <a href='https://findahelpline.com' style='color:#2ecc71;'>findahelpline.com</a></li>
    <li>🏥 <b>IASP Crisis Centres:</b> <a href='https://www.iasp.info/resources/Crisis_Centres/' style='color:#2ecc71;'>iasp.info</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Disclaimer ────────────────────────────────────────────────────────────────
def show_disclaimer():
    st.warning(
        "⚠️ **DISCLAIMER:** This is a **research prototype only** and is **NOT a clinical tool**. "
        "It must never replace professional mental health evaluation, diagnosis, or treatment. "
        "Predictions may be incorrect. Always consult a licensed professional."
    )


# ── Main App ──────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Text Screener")
st.markdown("*Suicidal Tendency Prediction from Social Media Text — Research Prototype*")

show_disclaimer()
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Enter Text")
    user_text = st.text_area(
        "Paste or type social media text below:",
        height=200,
        placeholder="Enter text here...",
        label_visibility="collapsed"
    )

    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not user_text.strip():
            st.error("Please enter some text.")
        else:
            try:
                tfidf, model, threshold = load_models()
                cleaned = preprocess(user_text)
                vec = tfidf.transform([cleaned])
                prob = model.predict_proba(vec)[0][1]
                pred = int(prob >= threshold)

                st.markdown("---")
                st.subheader("📊 Result")

                # Gauge chart
                fig, ax = plt.subplots(figsize=(6, 3))
                color = "#e74c3c" if pred == 1 else "#2ecc71"
                ax.barh(["Risk Score"], [prob], color=color, height=0.5)
                ax.barh(["Risk Score"], [1 - prob], left=[prob], color="#ecf0f1", height=0.5)
                ax.axvline(x=threshold, color="orange", linewidth=2, linestyle="--",
                           label=f"Threshold ({threshold:.2f})")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Risk Probability", fontweight="bold")
                ax.legend()
                st.pyplot(fig)

                st.metric("Risk Probability", f"{prob:.1%}")

                if pred == 1:
                    st.error("🔴 **HIGH RISK DETECTED** — Please refer to mental health resources immediately.")
                else:
                    st.success("🟢 **Low risk detected** — Resources are still available below.")

            except FileNotFoundError:
                st.error(
                    "Model files not found. Run the notebook first to generate and save:\n"
                    "- `tfidf_vectorizer.pkl`\n"
                    "- `lr_model.pkl`\n"
                    "- `lr_threshold.pkl`\n\n"
                    "Place them in the same folder as `app.py`."
                )

with col2:
    show_resources()

st.markdown("---")
st.caption(
    "EDGE Digital Skills Program | Applied Machine Learning | DUET, Gazipur | "
    "Research prototype — NOT for clinical use."
)
