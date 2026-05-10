# 🧠 Suicidal Tendency Prediction from Social Media Text

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aungchin203123-suicidalprediction.streamlit.app)

> ⚠️ **DISCLAIMER:** This is a **research prototype only** and is **NOT a clinical tool**. It must never replace professional mental health evaluation, diagnosis, or treatment.

---

## 📌 Project Overview

This project develops a machine learning pipeline to detect suicidal ideation in social media text using NLP techniques. Two models are trained and compared:

- **Logistic Regression** with TF-IDF features (deployed in GUI)
- **Bidirectional LSTM** deep learning model

Special focus is placed on **Recall optimisation** — a missed at-risk case is far more costly than a false alarm.

---

## 🚀 Live Demo

👉 **[https://aungchin203123-suicidalprediction.streamlit.app](https://aungchin203123-suicidalprediction.streamlit.app)**

---

## 📁 Project Structure

```
suicidal-prediction/
├── app.py                        # Streamlit GUI application
├── requirements.txt              # Python dependencies
├── tfidf_vectorizer.pkl          # Trained TF-IDF vectorizer
├── lr_model.pkl                  # Trained Logistic Regression model
├── lr_threshold.pkl              # Optimised classification threshold
├── bilstm_model.keras            # Trained BiLSTM model
├── lstm_tokenizer.json           # LSTM tokenizer
├── lstm_threshold.pkl            # BiLSTM threshold
└── suicidal-tendency-prediction.ipynb  # Full training notebook
```

---

## 🔧 Pipeline Summary

### 1. Preprocessing
- Lowercase, URL removal, Reddit pattern removal
- Contraction expansion (`won't` → `will not`)
- **Negation handling** — appends `_NEG` suffix to words following negation terms
- Tokenisation (NLTK punkt_tab), stopword removal (preserving `_NEG` tokens)

### 2. Feature Engineering
- TF-IDF vectorisation: unigrams + bigrams, 50,000 features
- Sublinear TF normalisation, min_df=2, max_df=0.95

### 3. Models
| Model | Features | Threshold |
|-------|----------|-----------|
| Logistic Regression | TF-IDF | F2-tuned |
| Bidirectional LSTM | Sequences (max_len=200) | F2-tuned |

### 4. Threshold Optimisation
F2-score maximisation on validation set (β=2 weights Recall 2× over Precision).

---

## 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.com/aungchin203123/suicidal-prediction.git
cd suicidal-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚖️ Ethics & Responsible AI

| Principle | Implementation |
|-----------|---------------|
| Transparency | Model type, data, and limits documented |
| Non-maleficence | Crisis resources shown regardless of prediction |
| Privacy | Zero data retention — no text stored |
| Human Oversight | Disclaimer + professional referral on every result |
| Fairness | Class-balanced training; bias limitations documented |

**Known Limitations:**
- English language only
- Reddit-trained — may not generalise to all demographics
- Cannot detect sarcasm or cultural nuances
- Not validated for clinical use


---

## 👨‍💻 Developer

**Aung Chin Wain**
- B.Sc. Mechanical Engineering — DUET, Gazipur
- CSWP & CSWA Certified | 3 International Conference Papers
- 🌐 [aungchinwain.github.io](https://aungchinwain.github.io/)

---

## 📋 Evaluation Criteria

| Criterion | Marks | Status |
|-----------|-------|--------|
| Preprocessing & Features | 20 | ✅ Negation handling, TF-IDF bigrams |
| Model + Threshold Tuning | 25 | ✅ LR + BiLSTM, F2-score tuning |
| Ethical Documentation | 20 | ✅ Ethics tab + README section |
| GUI + Safety Features | 20 | ✅ Disclaimer + resource panel |
| Report & Presentation | 15 | ✅ PDF report + live demo |
| **TOTAL** | **100** | 🎯 |

---

*© 2026 Aung Chin Wain | EDGE Digital Skills Program | Research prototype — NOT for clinical use.*
