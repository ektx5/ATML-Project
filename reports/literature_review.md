# Literature Review — Voice Clone Guard

## Status: Skeleton with [CITATION NEEDED] markers
## Fill in real citations before submission. Do not fabricate paper titles or authors.

---

## 1. The ASVspoof Challenge Series

The ASVspoof challenge is the primary benchmark for anti-spoofing research in speaker verification. Starting in 2015 and running through at least 2021, it defines standardised evaluation protocols, datasets, and metrics (Equal Error Rate, tandem-DCF) that allow direct comparison across systems.

### 1.1 Challenge Overview
- ASVspoof 2015 — first large-scale anti-spoofing challenge, TTS and VC attacks [CITATION NEEDED]
- ASVspoof 2017 — replay attack focus [CITATION NEEDED]
- ASVspoof 2019 — introduced Logical Access (LA) and Physical Access (PA) tracks [CITATION NEEDED]
  - LA track: synthetic speech from TTS and voice-conversion systems (A01–A19)
  - Class imbalance structure (~12K bonafide vs ~109K spoof in LA train) motivates class-weighted training
- ASVspoof 2021 — added deepfake track, codec-distorted conditions [CITATION NEEDED]

### 1.2 Evaluation Metrics
- Equal Error Rate (EER): the standard metric; rate at which false accept rate equals false reject rate [CITATION NEEDED]
- tandem-DCF: joint metric penalising both spoofing and speaker mismatch [CITATION NEEDED]
- Why accuracy alone is insufficient: a naive "always predict spoof" classifier achieves ~89% accuracy due to class imbalance, but EER = 50% [CITATION NEEDED]

---

## 2. wav2vec2-Based Spoof Detection

Wav2vec2.0 [CITATION NEEDED] is a self-supervised speech model trained on unlabelled audio via contrastive learning. Its feature representations capture fine-grained phoneme-level information.

### 2.1 Self-supervised Pre-training
- wav2vec2-base architecture: feature extractor CNN + 12-layer Transformer [CITATION NEEDED]
- Pre-trained on LibriSpeech 960h; features generalise to anti-spoofing without full retraining [CITATION NEEDED]

### 2.2 Application to Anti-Spoofing
- Fine-tuning wav2vec2 for anti-spoofing achieves state-of-the-art EER on ASVspoof 2019 LA [CITATION NEEDED]
- Partial fine-tuning (freeze lower layers, fine-tune upper layers) reduces catastrophic forgetting [CITATION NEEDED]
- Attention-pooling aggregation layer vs. mean pooling: attentional pooling improves EER [CITATION NEEDED]

### 2.3 Generalisation Gap
- Models trained on ASVspoof degrade significantly on out-of-distribution deepfakes [CITATION NEEDED]
  - Müller et al. In-the-Wild study [CITATION NEEDED]
  - Frank & Schönherr WaveFake study [CITATION NEEDED]
- Causes: over-fitting to codec/compression artefacts of ASVspoof's specific synthesis systems

---

## 3. Autoencoder-Based Anomaly Detection for Speech

### 3.1 Anomaly Detection via Reconstruction Error (ATML Unit 2)
- Autoencoders trained on "normal" data reconstruct normal samples well but fail on anomalies [CITATION NEEDED]
- Applied to speech: autoencoder trained on bonafide speech, reconstruction error signals spoofing [CITATION NEEDED]

### 3.2 Denoising Autoencoders
- Vincent et al. (2008) — denoising autoencoders are more robust representations than plain AEs [CITATION NEEDED]
- Applied to spectrogram reconstruction for voice authenticity [CITATION NEEDED]

### 3.3 Relationship to the GAN Framework (ATML Unit 3)
- GAN discriminator can be interpreted as a learned "naturalness" classifier [CITATION NEEDED]
- Reconstruction error provides an interpretable, single-number alternative that avoids GAN training instability [CITATION NEEDED]

---

## 4. Siamese Networks and Triplet Loss for Speaker Verification (ATML Unit 1)

### 4.1 Siamese Network Architecture
- Bromley et al. (1993) — original Siamese network for signature verification [CITATION NEEDED]
- Koch et al. (2015) — Siamese networks for one-shot image classification [CITATION NEEDED]
- Application to speaker verification: twin encoder with shared weights [CITATION NEEDED]

### 4.2 Metric Learning Losses
- Contrastive loss (Hadsell et al., 2006): penalises same-speaker embeddings that are far apart and different-speaker embeddings that are too close [CITATION NEEDED]
- Triplet loss (Schroff et al., 2015 — FaceNet): margin-based loss on (anchor, positive, negative) triplets [CITATION NEEDED]
- Semi-hard negative mining: avoids training collapse from trivially easy or pathologically hard negatives [CITATION NEEDED]

### 4.3 ECAPA-TDNN Speaker Encoder
- Desplanques et al. (2020) — ECAPA-TDNN: Emphasised Channel Attention, Propagation and Aggregation in TDNN [CITATION NEEDED]
- Multi-scale feature aggregation, SE blocks, and attentive statistics pooling [CITATION NEEDED]
- State-of-the-art on VoxCeleb1 and VoxCeleb2 speaker verification benchmarks [CITATION NEEDED]

### 4.4 Few-Shot Speaker Enrollment
- x-vectors (Snyder et al., 2018): d-vector embeddings for text-independent speaker verification [CITATION NEEDED]
- Enrollment by averaging N reference embeddings (zero-retraining few-shot) [CITATION NEEDED]
- Meta-learning perspective: prototypical networks (Snell et al., 2017) [CITATION NEEDED]

---

## 5. GAN-Based Vocoder Taxonomy (ATML Unit 3)

### 5.1 Neural Vocoders Used in ASVspoof Attacks
- WaveNet (van den Oord et al., 2016) — autoregressive waveform generation [CITATION NEEDED]
- WaveGlow (Prenger et al., 2019) — flow-based parallel synthesis [CITATION NEEDED]
- MelGAN (Kumar et al., 2019) — GAN-based mel-conditioned vocoder [CITATION NEEDED]
- HiFi-GAN (Kong et al., 2020) — multi-period discriminator GAN vocoder [CITATION NEEDED]
- Parallel WaveGAN (Yamamoto et al., 2020) [CITATION NEEDED]

### 5.2 Per-Vocoder Detection Difficulty
- WaveFake benchmark (Frank & Schönherr, 2021) — per-vocoder EER breakdown [CITATION NEEDED]
- GAN-based vocoders (MelGAN, HiFi-GAN) produce more natural-sounding speech and are harder to detect [CITATION NEEDED]
- Flow-based vocoders (WaveGlow) leave different spectral artefacts [CITATION NEEDED]

### 5.3 Arms Race Dynamics
- Anti-spoofing systems trained on older vocoders generalise poorly to newer ones [CITATION NEEDED]
- Relevance to ATML course: understanding GAN discriminator training explains why detection is challenging [CITATION NEEDED]

---

## 6. Attention Mechanisms in Speech Processing (ATML Unit 4)

### 6.1 Transformer Self-Attention
- Vaswani et al. (2017) — "Attention Is All You Need" [CITATION NEEDED]
- Application to speech: wav2vec2, Whisper, HuBERT all use Transformer encoder stacks [CITATION NEEDED]

### 6.2 Attention Pooling for Variable-Length Utterances
- x-vector statistics pooling: mean + std over frame-level features [CITATION NEEDED]
- Attentive statistics pooling (Okabe et al., 2018): learnable frame weights [CITATION NEEDED]
- Multi-head attention pooling: multiple attention heads capture different temporal patterns [CITATION NEEDED]

---

## 7. Ethics and Limitations

- Deepfake audio detection systems should output risk signals, not legal conclusions [CITATION NEEDED]
- Demographic bias in speaker verification: EER varies across accents, genders, age groups [CITATION NEEDED]
- Indian Standard Hour (ISH) alignment: SIH 2024 problem statement SIH26104 on voice cloning detection [CITATION NEEDED]
- GDPR / data protection implications of storing speaker enrollment embeddings [CITATION NEEDED]
