# Voice Clone Guard 🛡️

**AI-Powered Voice-Cloning Detection and Speaker-Verification System**  
*A Final-Year Engineering Project for Advanced Topics in Machine Learning (ATML)*

---

## 1. Project Overview

Voice Clone Guard addresses the critical dual challenge in modern biometric speech authentication:
1. **Spoof Detection (Branch A)**: Is the input speech authentic human speech or AI-synthesized (TTS / Voice Conversion / Neural Vocoder)?  
   *Syllabus mapping: ATML Unit 3 (GANs / Deep Generative Models) & Unit 4 (Self-Attention Mechanisms)*
2. **Speaker Verification (Branch B)**: Does the speech match the claimed enrolled identity?  
   *Syllabus mapping: ATML Unit 1 (Siamese Networks, Contrastive / Triplet Metric Learning, Few-Shot Meta-Learning)*
3. **Anomaly / Outlier Naturalness Signal**: Independent denoising autoencoder trained strictly on bonafide human speech spectrograms.  
   *Syllabus mapping: ATML Unit 2 (Autoencoder-based Anomaly / Outlier Detection)*
4. **Explainable Fusion**: Combines the three independent signals into a calibrated 0–100 risk score with plain-language diagnostic attribution.

> ⚠️ **Ethical & Operational Framing**: Voice Clone Guard outputs an explainable **risk signal for human-in-the-loop review**. It does not claim legal-grade mathematical proof of fraud.

---

## 2. Repository Layout

```
voice-clone-guard/
├── README.md                          # Project documentation & reproduction instructions
├── requirements.txt                   # Pinned pip requirements
├── environment.yml                    # Conda environment definition
├── .gitignore                         # Exclusions for weights, datasets, cache
├── config/
│   ├── branch_a_config.yaml           # Spoof detector hyperparameters
│   ├── branch_b_config.yaml           # Siamese speaker verification hyperparameters
│   └── fusion_config.yaml             # Fusion weights, paths & autoencoder settings
├── data/
│   ├── download_asvspoof.sh           # Download helper for ASVspoof 2019 LA
│   ├── download_voxceleb.sh           # Download helper for VoxCeleb1
│   ├── download_in_the_wild.sh        # Download helper for In-the-Wild (OOD benchmark)
│   └── README.md                      # Dataset license click-through instructions
├── src/
│   ├── datasets/
│   │   ├── asvspoof_dataset.py        # Protocol-driven PyTorch dataset & class weighting
│   │   ├── voxceleb_dataset.py        # Triplet / contrastive pair generator for VoxCeleb
│   │   └── audio_utils.py             # Resampling, VAD trimming, Log-Mel spectrograms, SpecAugment
│   ├── models/
│   │   ├── branch_a_spoof_detector.py # Pretrained wav2vec2 / CNN + manual Attention Pooling
│   │   ├── attention_pooling.py       # Custom Multi-Head Self-Attention Pooling (Unit 4 requirement)
│   │   ├── branch_b_siamese.py        # ECAPA-TDNN / ResNet-1D Siamese Twin Encoder
│   │   ├── denoising_autoencoder.py   # 2D Spectrogram Denoising Autoencoder (Unit 2 requirement)
│   │   └── fusion.py                  # Calibrated logistic & weighted sum risk fusion engine
│   ├── losses/
│   │   ├── contrastive_loss.py        # Contrastive margin loss
│   │   └── triplet_loss.py            # Triplet margin loss with online semi-hard negative mining
│   ├── train_branch_a.py              # Standalone CLI training script for Branch A
│   ├── train_branch_b.py              # Standalone CLI training script for Branch B
│   ├── train_autoencoder.py           # Bonafide-only autoencoder training script
│   ├── evaluate.py                    # Computes EER, AUC, F1, confusion matrices & OOD tables
│   ├── enroll_speaker.py              # Few-shot speaker enrollment CLI (averaging reference embeddings)
│   └── infer.py                       # End-to-end inference CLI (audio clip -> risk report)
├── notebooks/                         # EDA, DET curves, confusion analysis, and ablations
├── app/
│   ├── streamlit_demo.py              # Interactive Web UI (file upload, mic, risk gauges, explanations)
│   └── api.py                         # FastAPI REST service (/analyze, /enroll, /health)
├── tests/                             # Unit tests for core algorithms
└── reports/                           # Literature review, rubric outline, and exported CSV tables
```

---

## 3. Environment Setup

### Option A: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate voice-clone-guard
```

### Option B: Pip & Virtualenv
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Dataset Acquisition

1. **ASVspoof 2019 LA**: Follow instructions in `data/README.md` to register and verify checksums. Run `bash data/download_asvspoof.sh`.
2. **VoxCeleb1**: Register on the VGG portal, then run `bash data/download_voxceleb.sh`.
3. **In-the-Wild (Out-of-Domain)**: Run `bash data/download_in_the_wild.sh` to fetch the external test set. *Never train or tune on this split.*

---

## 5. Training Reproducibility

Train each component independently using deterministic seeds:

```bash
# 1. Train Spectrogram Denoising Autoencoder (Unit 2 Anomaly Detection)
python src/train_autoencoder.py --config config/fusion_config.yaml

# 2. Train Branch A Spoof Detector (wav2vec2 + Attention Pooling)
python src/train_branch_a.py --config config/branch_a_config.yaml

# 3. Train Branch B Speaker Verification (Siamese ECAPA-TDNN + Triplet Loss)
python src/train_branch_b.py --config config/branch_b_config.yaml
```

---

## 6. Evaluation & Benchmarking

Generate all experimental results and ablation tables:

```bash
# Full evaluation: in-domain ASVspoof eval, Out-of-Domain In-the-Wild, VoxCeleb verification, and ablations
python src/evaluate.py --config config/fusion_config.yaml --split all
```

The resulting CSVs and figures are saved directly to `reports/results_tables/`.

---

## 7. Interactive Applications

### Streamlit Web Application
```bash
streamlit run app/streamlit_demo.py
```

### FastAPI REST Service
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

---

## 8. Unit Tests

Run test suite:
```bash
pytest tests/ -v
```
