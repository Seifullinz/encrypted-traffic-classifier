#!/usr/bin/env python3
"""
app.py - Streamlit dashboard for encrypted traffic classification
=====================================================================
This is the demonstration "product": it presents the system and classifies
uploaded captures live.

Three tabs:
  1) Dataset Overview    - dataset composition and feature distributions
  2) Model Performance   - accuracy, confusion matrix, feature importance
  3) Live Classification - upload a .pcap and get a YouTube/Music verdict

Run with:
    streamlit run app.py

The dashboard reads the WINDOWED results (features_win.csv / results_win).
Run first:
    python3 feature_extraction.py --data-dir ./dataset --out features_win.csv --window 15
    python3 train_model.py --features features_win.csv --outdir results_win
=====================================================================
"""
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_extraction import extract_features_from_pcap

st.set_page_config(page_title="Encrypted Traffic Classifier",
                   page_icon="🛰️", layout="wide")

FEATURES_CSV = "features_win.csv"
MODEL_PATH = os.path.join("results_win", "model.joblib")
RESULTS_DIR = "results_win"


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_CSV) if os.path.exists(FEATURES_CSV) else None


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


st.title("🛰️ Encrypted Network Traffic Classifier")
st.caption("Flow-based YouTube vs Music classification (TLS 1.3 / QUIC) - "
           "no payload decryption. CSEC3100K Final Year Project.")

df = load_features()
bundle = load_model()

tab1, tab2, tab3 = st.tabs(
    ["📊 Dataset Overview", "🎯 Model Performance", "🔎 Live Classification"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Dataset Overview")
    if df is None:
        st.warning("features_win.csv not found. Run feature_extraction.py first.")
    else:
        feat_cols = [c for c in df.columns if c not in ("filename", "label", "source_file")]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total samples (windows)", len(df))
        c2.metric("Features", len(feat_cols))
        c3.metric("Classes", df["label"].nunique())

        st.markdown("**Class balance**")
        st.bar_chart(df["label"].value_counts())

        st.markdown("**Feature distribution by class** (choose a feature)")
        default = "downstream_byte_fraction"
        chosen = st.selectbox("Feature", feat_cols,
                              index=feat_cols.index(default) if default in feat_cols else 0)
        pivot = df.pivot(columns="label", values=chosen)
        st.bar_chart(pivot)

        with st.expander("Show raw feature table"):
            st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Model Performance")
    metrics_file = os.path.join(RESULTS_DIR, "metrics.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.warning("metrics.txt not found. Run train_model.py.")

    g1, g2 = st.columns(2)
    cm = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fi = os.path.join(RESULTS_DIR, "feature_importance.png")
    cv = os.path.join(RESULTS_DIR, "cv_scores.png")
    if os.path.exists(cm):
        g1.image(cm, caption="Confusion Matrix", use_container_width=True)
    if os.path.exists(cv):
        g2.image(cv, caption="5-Fold Cross-Validation", use_container_width=True)
    if os.path.exists(fi):
        st.image(fi, caption="Feature Importance", use_container_width=True)

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Live Classification")
    st.write("Upload a .pcap/.pcapng capture - the model extracts its features "
             "and classifies the traffic.")
    if bundle is None:
        st.warning("Model not found. Run train_model.py first.")
    else:
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        uploaded = st.file_uploader("PCAP file", type=["pcap", "pcapng"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Extracting features..."):
                    feats = extract_features_from_pcap(tmp_path)
                row = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
                pred = model.predict(row)[0]
                proba = model.predict_proba(row)[0]
                classes = list(model.classes_)

                st.success(f"### Prediction: **{pred}**")
                st.markdown("**Confidence**")
                st.bar_chart(pd.DataFrame({"probability": proba}, index=classes))

                with st.expander("Extracted features"):
                    st.json({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in feats.items()})
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not process the file: {e}")
            finally:
                os.unlink(tmp_path)
