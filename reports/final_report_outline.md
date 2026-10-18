# Final Report Outline — Voice Clone Guard
# ATML Course Project — Assessment Rubric Mapping

---

## 1. Domain Knowledge & Literature Review (Rubric Section 1)

### 1.1 Problem Statement
- AI voice cloning threat landscape (TTS, voice conversion, neural vocoders)
- SIH 2024 problem statement SIH26104 context
- Why existing phone-call / media-authentication systems fail
- Framing outputs as risk signals, not legal proof (ethics section)

### 1.2 Background
- The ASVspoof challenge series (2015, 2017, 2019, 2021)
- State-of-the-art deepfake detection models (wav2vec2, AASIST, RawNet2)
- State-of-the-art speaker verification (x-vector, d-vector, ECAPA-TDNN)

### 1.3 Literature Review
*(Reference literature_review.md — fill in actual citations)*
- ASVspoof challenge literature
- wav2vec2-based spoof detection
- Siamese/triplet-loss speaker verification
- GAN-based vocoder taxonomy (ATML Unit 3 connection)
- Attention mechanisms (ATML Unit 4 connection)
- Autoencoder anomaly detection (ATML Unit 2 connection)

### 1.4 Gap Analysis
- Generalisation gap: ASVspoof vs. real-world (In-the-Wild)
- Multi-signal fusion: why single-branch systems are insufficient
- Few-shot enrollment: practical deployment requirement

---

## 2. Implementation & Performance Evaluation (Rubric Section 2)

### 2.1 System Architecture
- Figure: end-to-end block diagram (input -> Branch A | Branch B | AE -> Fusion -> output)
- ATML unit mapping:
  - Unit 1: Siamese + triplet loss (Branch B)
  - Unit 2: Denoising autoencoder anomaly detection
  - Unit 3: GAN taxonomy, vocoder-family detection
  - Unit 4: Multi-head attention pooling (hand-implemented, Branch A)

### 2.2 Implementation Details
- Branch A: wav2vec2-base, freeze 6 layers, multi-head attention pooling
- Branch B: ECAPA-TDNN twin encoder, triplet loss, semi-hard mining
- Autoencoder: 2-D conv encoder-decoder on Mel spectrograms, bonafide-only training
- Fusion: logistic regression over 3 signals

### 2.3 Training Protocol
- Seeds, determinism
- Datasets and preprocessing
- Class imbalance handling (class-weighted CE)
- Hyperparameters (refer to config/*.yaml)

### 2.4 Results

#### Table 1 — Branch A: ASVspoof 2019 LA Evaluation Set
| Model | EER (%) | AUC | F1 |
|-------|---------|-----|----|
| Baseline (CNN, no attention pooling) | _TBD_ | _TBD_ | _TBD_ |
| Branch A (wav2vec2 + attention pooling) | _TBD_ | _TBD_ | _TBD_ |

#### Table 2 — Generalisation Gap (OOD Evaluation) ← KEY RESULT
| Evaluation Split | EER (%) | AUC | F1 |
|-----------------|---------|-----|----|
| ASVspoof 2019 LA eval (in-domain) | _TBD_ | _TBD_ | _TBD_ |
| In-the-Wild (out-of-domain) | _TBD_ | _TBD_ | _TBD_ |
| **Generalisation gap (Δ EER)** | _TBD_ | — | — |

*Note: The gap between in-domain and OOD EER is expected to be significant (often 10–30 pp). Discuss this honestly even if the numbers are unflattering — the gap itself is scientifically interesting.*

#### Table 3 — Branch A: Per-Attack-Type EER Breakdown
| Attack ID | Type | EER (%) |
|-----------|------|---------|
| A01 | TTS-waveform | _TBD_ |
| A02 | TTS-waveform | _TBD_ |
| ... | ... | ... |
| A19 | VC | _TBD_ |

#### Table 4 — Branch B: Speaker Verification EER (VoxCeleb1)
| System | EER (%) | AUC |
|--------|---------|-----|
| Branch B (ECAPA-TDNN, triplet loss) | _TBD_ | _TBD_ |
| Branch B (ECAPA-TDNN, contrastive loss) | _TBD_ | _TBD_ |

#### Table 5 — Ablation Study: Fusion Signal Contributions ← KEY RESULT
| System Configuration | EER (%) | AUC |
|----------------------|---------|-----|
| Full system (all 3 signals) | _TBD_ | _TBD_ |
| Without AE signal (spoof + speaker only) | _TBD_ | _TBD_ |
| Without Branch B (spoof + AE only) | _TBD_ | _TBD_ |
| Without Branch A (AE + speaker only) | _TBD_ | _TBD_ |

*This table quantifies how much each branch contributes to the final decision.*

#### Table 6 — Optional: WaveFake Per-Vocoder Breakdown
| Vocoder | EER (%) |
|---------|---------|
| MelGAN | _TBD_ |
| HiFi-GAN | _TBD_ |
| WaveGlow | _TBD_ |
| Parallel WaveGAN | _TBD_ |

### 2.5 Error Analysis
- Which attack types / speakers are hardest to detect?
- Confusion matrix interpretation
- Example failure cases (false negatives — missed spoofs)

---

## 3. Research Paper (Rubric Section 3)

### 3.1 Abstract
*(~200 words — write after getting final results)*

### 3.2 Introduction
- Motivation: AI voice cloning threat
- Our contribution: multi-signal fusion system
- Paper structure overview

### 3.3 Related Work
*(Summarise literature_review.md)*

### 3.4 System Design
*(Reference Section 2.1 above)*

### 3.5 Experiments
*(Reference Tables 1–6 above)*

### 3.6 Discussion
- What drives the generalisation gap?
- When does the AE signal add value beyond Branch A?
- Few-shot enrollment analysis: how many clips needed for reliable verification?

### 3.7 Ethics & Limitations
- Risk-signal framing: outputs are NOT legal-grade fraud proof
- Demographic bias in speaker verification
- Data memorisation risks in enrollment store
- Environmental cost of training large speech models

### 3.8 Conclusion
- Summary of findings
- Future work: adversarial training for robustness, continual learning for new vocoder types

### 3.9 References
*(Fill in from literature_review.md — all [CITATION NEEDED] markers must be resolved)*

---

## Appendix A — Ablation Study Design Template

```
Experiment: Signal Contribution Ablation
Dataset: ASVspoof 2019 LA eval + In-the-Wild
Metric: EER (primary), AUC-ROC (secondary)

For each ablation condition:
  1. Load all trained models (Branch A, Branch B, AE)
  2. Zero out / replace the ablated signal:
     - No AE:       ae_error_norm = 0.0
     - No Branch B: speaker_similarity = 0.0
     - No Branch A: spoof_prob = 0.5  (chance)
  3. Re-run fusion on the full eval set
  4. Compute EER and AUC
  5. Report in Table 5

Command to reproduce:
  python src/evaluate.py --config config/fusion_config.yaml --split ablation
```

## Appendix B — Experiment Reproducibility Checklist

- [ ] Random seeds set: numpy, torch, python random (seed=42)
- [ ] Class weights computed from training split, not hardcoded
- [ ] EER threshold tuned on validation set, not hardcoded
- [ ] In-the-Wild dataset NEVER touched during training or validation
- [ ] Official VoxCeleb1 veri_test.txt used for reported EER (not random pairs)
- [ ] All training commands in README.md reproduce the reported numbers
- [ ] All evaluation tables generated by `python src/evaluate.py` (not hand-entered)
- [ ] Checkpoints include epoch, val_eer, and random state for resumability
