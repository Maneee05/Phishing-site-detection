# app.py
import streamlit as st
import joblib
import numpy as np
from urllib.parse import urlparse
import re
import tldextract

# ---------- CONFIG ----------
#MODEL_PATH = "phishing_rf_model.pkl"  # ensure this file is inside project root
MODEL_PATH = "phishing-streamlit/phishingdetection_py.py"
st.set_page_config(page_title="Phishing Detector", layout="centered")

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    return model

loaded_model = load_model()

# ---------- FEATURE EXTRACTOR ----------
# IMPORTANT: This extractor MUST match the features & ordering used during training.
def is_ip_in_domain(url):
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return 0
        return 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0
    except:
        return 0

def count_digits(s):
    return sum(c.isdigit() for c in s)

def has_https(url):
    return 1 if url.lower().startswith("https://") else 0

def count_char(url, ch):
    return url.count(ch)

def get_tld_length(url):
    try:
        ext = tldextract.extract(url)
        return len(ext.suffix) if ext.suffix else 0
    except:
        return 0

def suspicious_words_count(url):
    suspicious = ["login", "verify", "update", "bank", "secure", "account", "confirm"]
    s = url.lower()
    return sum(1 for w in suspicious if w in s)

def extract_features_from_url(url):
    """
    Return numpy array of features in the SAME ORDER used at training.
    Example order used here (change if your training used different columns):
      [url_length, https, num_dots, num_hyphens, num_at,
       ip_in_domain, num_digits, tld_length, suspicious_words, has_www]
    """
    url_str = str(url).strip()
    url_length = len(url_str)
    https = has_https(url_str)
    num_dots = count_char(url_str, ".")
    num_hyphens = count_char(url_str, "-")
    num_at = count_char(url_str, "@")
    ip_in_domain = is_ip_in_domain(url_str)
    digits = count_digits(url_str)
    tld_len = get_tld_length(url_str)
    susp_words = suspicious_words_count(url_str)
    has_www = 1 if "www." in url_str.lower() else 0

    feature_vector = [
        url_length,
        https,
        num_dots,
        num_hyphens,
        num_at,
        ip_in_domain,
        digits,
        tld_len,
        susp_words,
        has_www
    ]
    return np.array(feature_vector, dtype=float)

# ---------- PREDICTION HELPERS ----------
def predict_from_url(url):
    fv = extract_features_from_url(url).reshape(1, -1)
    pred = loaded_model.predict(fv)[0]
    try:
        conf = loaded_model.predict_proba(fv).max()
    except Exception:
        conf = None
    label = "Phishing" if int(pred) == 1 else "Safe"
    return {"label": label, "pred_value": int(pred), "confidence": float(conf) if conf is not None else None, "features": fv.flatten().tolist()}

# ---------- STREAMLIT UI ----------
st.title("🔎 Phishing Website Detector")
st.write("Enter a URL and the model will predict whether it's likely phishing or safe. Make sure your model file `phishing_rf_model.pkl` is in the same folder as this app.")

url_input = st.text_input("Enter website URL", placeholder="https://example.com/login")
col1, col2 = st.columns([1,3])
with col1:
    if st.button("Predict"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Extracting features and predicting..."):
                try:
                    res = predict_from_url(url_input.strip())
                    if res["label"] == "Phishing":
                        st.error(f'⚠️ Prediction: {res["label"]}')
                    else:
                        st.success(f'✅ Prediction: {res["label"]}')
                    if res["confidence"] is not None:
                        st.write(f"Confidence: **{res['confidence']*100:.1f}%**")
                    st.write("Features used (in order):")
                    st.write(res["features"])
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

with col2:
    st.info("Tips:\n- This demo uses string-based feature extraction (no page fetch). \n- Do not visit suspicious URLs in your browser. \n- Ensure extractor ordering matches training features.")

st.markdown("---")
st.subheader("Test safely")
st.write("Use reserved domains like `example.com` or `.test` in your manual tests.")
