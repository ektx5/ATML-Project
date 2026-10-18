"""
streamlit_demo.py — Interactive Web Application for Voice Clone Guard.

Usage
-----
    streamlit run app/streamlit_demo.py
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import load_all_models, run_inference
from src.enroll_speaker import load_enrollment_store, save_enrollment_store
from src.datasets.audio_utils import load_and_preprocess
from src.models.branch_b_siamese import enroll_speaker

st.set_page_config(
    page_title="Voice Clone Guard 🛡️",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_PATH = "config/fusion_config.yaml"


@st.cache_resource(show_spinner="Loading deep learning models...")
def get_cached_models():
    if not os.path.exists(CONFIG_PATH):
        return None, None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        fusion_cfg = yaml.safe_load(f)
    device = torch.device("cuda" if fusion_cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    models = load_all_models(fusion_cfg, device)
    return models, fusion_cfg, device


def render_gauge(risk_score: float) -> None:
    try:
        import plotly.graph_objects as go
        if risk_score < 30.0:
            color = "#28a745"  # Green
        elif risk_score < 60.0:
            color = "#ffc107"  # Yellow/Orange
        else:
            color = "#dc3545"  # Red

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Overall Risk Score", "font": {"size": 22, "color": "#333"}},
            number={"suffix": " / 100", "font": {"size": 32, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 60], "color": "#fffde7"},
                    {"range": [60, 100], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": risk_score,
                },
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.metric(label="Risk Score", value=f"{risk_score:.1f} / 100")
        st.progress(int(np.clip(risk_score, 0, 100)))


def render_explainability(result: Dict) -> None:
    st.subheader("🔍 Explainable Risk Attribution")

    exp = result.get("explanation", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="AI Deepfake / Spoof Signal",
            value=f"{exp.get('spoof_percentage', 0.0)}%",
            delta=f"P(spoof) = {result.get('spoof_probability', 0.0):.3f}",
            delta_color="inverse",
            help="Deep neural network + multi-head attention pooling detecting synthetic speech artefacts."
        )

    with col2:
        st.metric(
            label="Acoustic Naturalness Anomaly",
            value=f"{exp.get('ae_percentage', 0.0)}%",
            delta=f"z-score = {result.get('ae_error_normalised', 0.0):.2f}",
            delta_color="inverse",
            help="Reconstruction MSE on log-Mel spectrogram compared against bonafide speech distribution."
        )

    with col3:
        claimed = result.get("claimed_speaker_id")
        sim_val = result.get("speaker_similarity", 0.0)
        st.metric(
            label="Speaker Identity Match",
            value=f"{exp.get('speaker_percentage', 0.0)}%",
            delta=f"Similarity = {sim_val:.3f}" if claimed else "No claimed ID",
            delta_color="normal",
            help="Siamese ECAPA-TDNN twin encoder cosine match against enrolled voiceprint."
        )


def main():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=Outfit:wght@600;700;800&display=swap');
        .custom-main-title {
            font-family: 'Plus Jakarta Sans', 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 2.7rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: #0F172A;
            margin-bottom: 0.5rem;
            padding-top: 0.2rem;
        }
        </style>
        <h1 class="custom-main-title">🛡️ Voice Clone Guard</h1>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    models_data = get_cached_models()
    if models_data[0] is None:
        st.error(f"Configuration file not found at {CONFIG_PATH}. Please check repository setup.")
        return

    models, fusion_cfg, device = models_data
    model_a, model_b, ae, ae_stats, fusion_engine, branch_a_cfg, branch_b_cfg = models

    store_path = fusion_cfg["branch_b"].get("enrollment_store", "checkpoints/branch_b/enrollments.pkl")
    store = load_enrollment_store(store_path)

    # Sidebar: Few-shot Speaker Enrollment
    st.sidebar.header("👤 Few-Shot Speaker Enrollment")
    st.sidebar.markdown(
        "Register a legitimate speaker identity from **3 to 5 reference clips** "
        "(zero retraining, embedding-space meta-learning)."
    )

    enroll_id = st.sidebar.text_input("Claimed Speaker Name / ID", placeholder="e.g. alice")
    ref_files = st.sidebar.file_uploader(
        "Upload 3–5 reference audio files",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
        key="ref_uploader",
    )

    if st.sidebar.button("Enroll Voice Identity", type="primary", disabled=not (enroll_id and ref_files)):
        if len(ref_files) < 3 or len(ref_files) > 5:
            st.sidebar.warning(f"Please provide between 3 and 5 reference clips (got {len(ref_files)}).")
        else:
            with st.sidebar.status("Processing reference clips..."):
                mel_specs = []
                for f in ref_files:
                    suffix = Path(f.name).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        tmp_p = tmp.name
                    try:
                        mel = load_and_preprocess(tmp_p, branch_b_cfg["audio"], augment=False)
                        mel_specs.append(mel)
                    finally:
                        if os.path.exists(tmp_p):
                            os.unlink(tmp_p)

                avg_emb = enroll_speaker(model_b, mel_specs, device)
                store[enroll_id] = avg_emb.cpu()
                save_enrollment_store(store, store_path)
            st.sidebar.success(f"Enrolled identity '{enroll_id}' successfully!")
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("**Enrolled Voices:**")
    if store:
        for s_id in list(store.keys()):
            sc1, sc2 = st.sidebar.columns([3, 1])
            sc1.write(f"• {s_id}")
            if sc2.button("🗑️", key=f"del_{s_id}"):
                del store[s_id]
                save_enrollment_store(store, store_path)
                st.rerun()
    else:
        st.sidebar.caption("No speakers enrolled yet.")

    # Main Area: Audio Analysis
    st.header("🔊 Test Audio Verification")

    col_input, col_meta = st.columns([2, 1])
    with col_input:
        uploaded_clip = st.file_uploader("Upload test audio file (WAV / MP3 / FLAC)", type=["wav", "mp3", "flac"])
        mic_audio = st.audio_input("Or record speech directly:")

    with col_meta:
        enrolled_options = ["(None / Unclaimed)"] + list(store.keys())
        selected_spk = st.selectbox("Claimed Speaker Identity", options=enrolled_options)
        claimed_id = None if selected_spk == "(None / Unclaimed)" else selected_spk

    audio_bytes = None
    if mic_audio is not None:
        audio_bytes = mic_audio.read()
        st.audio(audio_bytes, format="audio/wav")
    elif uploaded_clip is not None:
        audio_bytes = uploaded_clip.read()
        st.audio(audio_bytes)

    if st.button("🚀 Analyze Voice Authenticity", type="primary", disabled=(audio_bytes is None)):
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with st.spinner("Executing multi-branch biometric verification..."):
            try:
                result = run_inference(
                    audio_path=tmp_path,
                    fusion_cfg=fusion_cfg,
                    device=device,
                    speaker_id=claimed_id,
                    preloaded_models=models,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        st.divider()

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            render_gauge(result["risk_score"])
        with res_col2:
            verdict = result["verdict"]
            sim = result["speaker_similarity"]
            p_spoof = result["spoof_probability"]

            if "LOW" in verdict:
                st.success(f"### Verdict: {verdict}")
                st.markdown("✅ **Authentic Voice Verified.** Low likelihood of voice cloning or speaker mismatch.")
            elif "MEDIUM" in verdict:
                st.warning(f"### Verdict: {verdict}")
                st.markdown("⚠️ **Borderline Threat Level.** Recommended for human analyst inspection.")
            else:
                st.error(f"### Verdict: {verdict}")
                st.markdown("🚨 **High Threat Detected.** Flagged as potential AI clone or impostor.")

            if claimed_id:
                if sim >= 0.70:
                    st.success(f"👤 **Speaker Identity Confirmed**: **{sim*100:.1f}% Match** with enrolled voice **'{claimed_id}'**.")
                else:
                    st.error(f"👤 **Speaker Mismatch**: Only **{sim*100:.1f}% Match** with claimed voice **'{claimed_id}'**.")
            else:
                st.info("👤 No claimed identity selected. Running unconstrained spoof analysis.")

        st.divider()
        render_explainability(result)


if __name__ == "__main__":
    main()
