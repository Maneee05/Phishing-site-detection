"""
app.py  –  Phishing Website Detector  (Streamlit)
==================================================
Run:  streamlit run app.py
"""

import re, json, os, math
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import tldextract

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard – Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark theme, glassmorphism, animations)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(22,27,34,0.95);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(120deg, #1a1f2e 0%, #0f2942 50%, #1a1f2e 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    text-align: center;
}
.hero-banner h1 {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #63b3ed, #90cdf4, #63b3ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-banner p {
    color: #8b949e;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Card ── */
.card {
    background: rgba(22,27,34,0.85);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Result badges ── */
.result-safe {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.result-phishing {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.result-safe h2, .result-phishing h2 {
    font-size: 2rem;
    font-weight: 900;
    margin: 0.3rem 0;
}
.result-safe p, .result-phishing p { margin: 0; color: rgba(255,255,255,0.8); }

/* ── Feature row ── */
.feat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0.8rem;
    border-radius: 8px;
    margin: 3px 0;
    font-size: 0.88rem;
}
.feat-row:nth-child(odd)  { background: rgba(255,255,255,0.03); }
.feat-bad   { color: #fc8181; font-weight: 600; }
.feat-good  { color: #68d391; font-weight: 600; }
.feat-neutral { color: #a0aec0; }

/* ── Metric box ── */
.metric-box {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    text-align: center;
}
.metric-box .val { font-size: 1.6rem; font-weight: 800; color: #63b3ed; }
.metric-box .lbl { font-size: 0.78rem; color: #8b949e; margin-top: 2px; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Input ── */
input[type="text"] {
    background: #161b22 !important;
    border: 1.5px solid rgba(99,179,237,0.3) !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 1rem !important;
    padding: 0.6rem !important;
}
input[type="text"]:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
}

/* ── Hide Streamlit footer ── */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
# Fall back to the old compressed model if new one not trained yet
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "phishing_rf_model_comprsd.pkl")

FEAT_PATH  = os.path.join(BASE_DIR, "feature_names.json")

# The 30 features from the Kaggle Phishing Websites Dataset
FEATURES_30 = [
    'UsingIP','LongURL','ShortURL','Symbol@','Redirecting//',
    'PrefixSuffix-','SubDomains','HTTPS','DomainRegLen','Favicon',
    'NonStdPort','HTTPSDomainURL','RequestURL','AnchorURL',
    'LinksInScriptTags','ServerFormHandler','InfoEmail','AbnormalURL',
    'WebsiteForwarding','StatusBarCust','DisableRightClick',
    'UsingPopupWindow','IframeRedirection','AgeofDomain','DNSRecording',
    'WebsiteTraffic','PageRank','GoogleIndex','LinksPointingToPage','StatsReport'
]

FEATURE_DESCRIPTIONS = {
    'UsingIP':            'IP address used as domain',
    'LongURL':            'URL length > 54 chars',
    'ShortURL':           'URL shortening service used',
    'Symbol@':            '@ symbol in URL',
    'Redirecting//':      'Double-slash redirect in path',
    'PrefixSuffix-':      'Hyphen in domain name',
    'SubDomains':         'Multiple subdomains',
    'HTTPS':              'HTTPS protocol used',
    'DomainRegLen':       'Domain registration length',
    'Favicon':            'Favicon from external domain',
    'NonStdPort':         'Non-standard port used',
    'HTTPSDomainURL':     'HTTPS token in domain',
    'RequestURL':         'External objects loaded',
    'AnchorURL':          'Suspicious anchor tags',
    'LinksInScriptTags':  'Links in script/meta tags',
    'ServerFormHandler':  'Server form handler suspicious',
    'InfoEmail':          'Email address in URL',
    'AbnormalURL':        'URL not matching hostname',
    'WebsiteForwarding':  'Excessive redirects',
    'StatusBarCust':      'Status bar customisation',
    'DisableRightClick':  'Right-click disabled',
    'UsingPopupWindow':   'Popup windows used',
    'IframeRedirection':  'iFrame redirect present',
    'AgeofDomain':        'Domain age < 6 months',
    'DNSRecording':       'DNS record missing',
    'WebsiteTraffic':     'Low website traffic',
    'PageRank':           'Low Google PageRank',
    'GoogleIndex':        'Not indexed by Google',
    'LinksPointingToPage':'Few links pointing to page',
    'StatsReport':        'Flagged in stats reports',
}

SHORTENERS = re.compile(
    r"bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|"
    r"buff\.ly|adf\.ly|short\.link|rb\.gy|shorte\.st"
)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (30 features, same order as training dataset)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(url: str) -> dict:
    """Return dict {feature_name: int} with values -1 (phishing), 0 (neutral), 1 (safe)."""
    url = url.strip()
    try:
        parsed   = urlparse(url if "://" in url else "http://" + url)
        ext      = tldextract.extract(url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        netloc   = parsed.netloc or ""
        scheme   = parsed.scheme.lower()
    except Exception:
        return {f: 0 for f in FEATURES_30}

    f = {}

    # 1. UsingIP – IP address as domain → phishing=1, safe=-1
    f['UsingIP'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else -1

    # 2. LongURL – len>75 phishing, 54-75 neutral, <54 safe
    l = len(url)
    f['LongURL'] = 1 if l > 75 else (0 if l >= 54 else -1)

    # 3. ShortURL – shortening service
    f['ShortURL'] = 1 if SHORTENERS.search(url) else -1

    # 4. Symbol@ – @ in URL
    f['Symbol@'] = 1 if '@' in url else -1

    # 5. Redirecting// – // after position 7
    f['Redirecting//'] = 1 if '//' in path else -1

    # 6. PrefixSuffix- – hyphen in domain
    f['PrefixSuffix-'] = 1 if '-' in (ext.domain + ext.subdomain) else -1

    # 7. SubDomains – count dots in hostname
    dots = hostname.count('.')
    f['SubDomains'] = -1 if dots <= 1 else (0 if dots == 2 else 1)

    # 8. HTTPS – safe if HTTPS
    f['HTTPS'] = -1 if scheme == 'https' else 1

    # 9. DomainRegLen – heuristic: short domains often phishing
    domain_len = len(ext.domain)
    f['DomainRegLen'] = 1 if domain_len < 5 else -1

    # 10. Favicon – heuristic: placeholder safe
    f['Favicon'] = -1

    # 11. NonStdPort
    port = parsed.port
    std_ports = {80, 443, 8080, 8443, None}
    f['NonStdPort'] = 1 if port not in std_ports else -1

    # 12. HTTPSDomainURL – 'https' token in domain name
    f['HTTPSDomainURL'] = 1 if 'https' in hostname.lower() else -1

    # 13. RequestURL – heuristic (can't fetch page)
    f['RequestURL'] = -1

    # 14. AnchorURL – heuristic
    f['AnchorURL'] = -1

    # 15. LinksInScriptTags – heuristic
    f['LinksInScriptTags'] = -1

    # 16. ServerFormHandler – heuristic
    f['ServerFormHandler'] = -1

    # 17. InfoEmail – mailto in URL
    f['InfoEmail'] = 1 if 'mailto:' in url.lower() else -1

    # 18. AbnormalURL – hostname not in URL or suspicious chars
    f['AbnormalURL'] = 1 if (hostname and hostname not in url) else -1

    # 19. WebsiteForwarding – heuristic
    f['WebsiteForwarding'] = -1

    # 20-22. Behaviour features – heuristic defaults (safe)
    f['StatusBarCust']   = -1
    f['DisableRightClick'] = -1
    f['UsingPopupWindow']  = -1

    # 23. IframeRedirection
    f['IframeRedirection'] = -1

    # 24. AgeofDomain – heuristic: unknown → suspect
    f['AgeofDomain'] = 0

    # 25. DNSRecording – heuristic
    f['DNSRecording'] = 0

    # 26. WebsiteTraffic – heuristic
    f['WebsiteTraffic'] = 0

    # 27. PageRank – heuristic
    f['PageRank'] = 0

    # 28. GoogleIndex – heuristic
    f['GoogleIndex'] = -1

    # 29. LinksPointingToPage – heuristic
    f['LinksPointingToPage'] = 0

    # 30. StatsReport
    f['StatsReport'] = -1

    # Extra heuristics that boost suspicion score
    susp_words = ['login','verify','update','bank','secure','account',
                  'confirm','paypal','ebay','amazon','apple','microsoft',
                  'password','signin','credential','wallet']
    hits = sum(1 for w in susp_words if w in url.lower())
    if hits >= 2:
        f['AbnormalURL'] = 1
        f['StatsReport'] = 1
    if hits >= 3:
        f['ServerFormHandler'] = 1

    query = parsed.query or ""
    if len(query) > 100:
        f['RequestURL'] = 1
    if url.count('/') > 6:
        f['WebsiteForwarding'] = 1

    return f

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(url: str):
    model = load_model()
    feat_dict = extract_features(url)

    # Decide feature order
    if os.path.exists(FEAT_PATH):
        with open(FEAT_PATH) as fh:
            cols = json.load(fh)
    else:
        cols = FEATURES_30

    vec = np.array([feat_dict.get(c, 0) for c in cols], dtype=float).reshape(1, -1)

    pred = int(model.predict(vec)[0])
    try:
        proba = model.predict_proba(vec)[0]
        conf  = float(proba[pred])
    except Exception:
        conf = None

    label = "Phishing" if pred == 1 else "Safe"
    return label, conf, feat_dict, cols

# ─────────────────────────────────────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(conf: float, label: str) -> go.Figure:
    color = "#ef4444" if label == "Phishing" else "#10b981"
    val   = conf * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0,  40], "color": "rgba(16,185,129,0.15)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": val,
            },
        },
        title={"text": f"Confidence — {label}",
               "font": {"color": "#8b949e", "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=40, b=0, l=20, r=20),
        font={"family": "Inter"},
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ PhishGuard")
    st.markdown("---")
    st.markdown("""
**How it works**

1. Enter any website URL  
2. The app extracts **30 security features**  
3. A trained **Random Forest / Ensemble** model predicts risk  
4. You see a confidence score + feature breakdown  
""")
    st.markdown("---")
    st.markdown("**⚙️ Model Info**")
    model_name = "Ensemble (RF + GB)" if "phishing_model" in MODEL_PATH else "Random Forest"
    st.info(f"Model: {model_name}\n\nFeatures: 30 URL-based\n\nDataset: Kaggle Phishing Websites")
    st.markdown("---")
    st.markdown("**🔒 Safe test URLs**")
    st.code("https://www.google.com\nhttps://www.github.com\nhttps://www.microsoft.com")
    st.markdown("**⚠️ Phishing patterns**")
    st.code("http://192.168.1.1/login\nhttp://paypal-verify.tk/secure\nhttp://bit.ly/xfakelogin")
    st.markdown("---")
    st.caption("⚡ No page visited — analysis is URL-only.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🛡️ PhishGuard</h1>
  <p>Detect phishing &amp; unsafe websites instantly using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Input row
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com/login",
        label_visibility="collapsed",
    )
with col_btn:
    predict_clicked = st.button("🔍 Analyse", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Prediction ───────────────────────────────────────────────────────────────
if predict_clicked:
    if not url_input.strip():
        st.warning("⚠️ Please enter a URL first.")
    else:
        with st.spinner("Extracting features and running model…"):
            try:
                label, conf, feat_dict, cols = predict(url_input.strip())
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # ── Result badge
        icon  = "⚠️" if label == "Phishing" else "✅"
        cls   = "result-phishing" if label == "Phishing" else "result-safe"
        color = "#fc8181" if label == "Phishing" else "#68d391"
        st.markdown(f"""
<div class="{cls}">
  <div style="font-size:2.5rem">{icon}</div>
  <h2 style="color:{color}">{label} Site Detected</h2>
  <p>{'This URL shows signs of a phishing attack. Do NOT visit it.' if label=='Phishing' else 'No strong phishing indicators found in this URL.'}</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence gauge + quick metrics
        g_col, m_col = st.columns([1, 1])
        with g_col:
            if conf is not None:
                st.plotly_chart(make_gauge(conf, label),
                                use_container_width=True, config={"displayModeBar": False})
        with m_col:
            susp = sum(1 for v in feat_dict.values() if v == 1)
            safe_cnt = sum(1 for v in feat_dict.values() if v == -1)
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"""<div class="metric-box">
<div class="val">{susp}</div>
<div class="lbl">🔴 Risky Signals</div>
</div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div class="metric-box">
<div class="val">{safe_cnt}</div>
<div class="lbl">🟢 Safe Signals</div>
</div>""", unsafe_allow_html=True)
            with mc3:
                pct = f"{conf*100:.1f}%" if conf else "N/A"
                st.markdown(f"""<div class="metric-box">
<div class="val">{pct}</div>
<div class="lbl">🎯 Confidence</div>
</div>""", unsafe_allow_html=True)

        # ── Feature breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Feature Breakdown (all 30 signals)", expanded=True):
            rows_html = ""
            for feat in FEATURES_30:
                val  = feat_dict.get(feat, 0)
                desc = FEATURE_DESCRIPTIONS.get(feat, feat)
                if val == 1:
                    badge = '<span class="feat-bad">⚠ Risky</span>'
                    cls2  = "feat-bad"
                elif val == -1:
                    badge = '<span class="feat-good">✓ OK</span>'
                    cls2  = "feat-good"
                else:
                    badge = '<span class="feat-neutral">– Neutral</span>'
                    cls2  = "feat-neutral"
                rows_html += f"""
<div class="feat-row">
  <span class="{cls2}">{feat}</span>
  <span style="color:#8b949e;font-size:0.82rem">{desc}</span>
  {badge}
</div>"""
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

        # ── Session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, {
            "URL": url_input.strip()[:60] + ("…" if len(url_input) > 60 else ""),
            "Result": label,
            "Confidence": f"{conf*100:.1f}%" if conf else "N/A",
            "Risky Signals": susp,
        })
        st.session_state.history = st.session_state.history[:10]

# ── History table
if "history" in st.session_state and st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕑 Recent Predictions")
    hist_df = pd.DataFrame(st.session_state.history)
    def color_result(val):
        return "color: #fc8181; font-weight:bold" if val == "Phishing" else "color: #68d391; font-weight:bold"
    styled = hist_df.style.applymap(color_result, subset=["Result"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Footer
st.markdown("""
<br><br>
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 1rem 0; border-top: 1px solid rgba(255,255,255,0.05);">
  🛡️ PhishGuard &nbsp;|&nbsp; Built with scikit-learn, Streamlit &amp; Plotly &nbsp;|&nbsp; URL-only analysis — no page visit required
</div>
""", unsafe_allow_html=True)
