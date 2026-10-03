---
title: ClarityMed AI
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.27.0
python_version: "3.13"
app_file: app.py
pinned: false
---

# ClarityMed AI

Clinical decision-support research prototype that **shows how sure its models really are**.
It combines three analysis modules in one interface and produces a single PDF report for the session.

> **Research prototype. Not a diagnostic system.** Do not enter real patient information.
> Model scores are raw outputs, not calibrated probabilities.

## Modules

| Module | Model | Data |
|---|---|---|
| Chest X-ray (NORMAL / PNEUMONIA) | DenseNet-121 (ImageNet transfer learning, class-weighted loss) + Grad-CAM | Public Kaggle chest X-ray pneumonia dataset |
| Brain MRI (glioma / meningioma / pituitary / no tumor) | ResNet-50 (transfer learning) + Grad-CAM | Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar) |
| Symptom checker (24 conditions) | TF-IDF + Logistic Regression | Symptom2Disease (1,200 text samples) |

An optional Gemini call rewrites the result in plain language. It is skipped when no API key is set.

## What the interface does with uncertainty

- Every result shows the raw model score with a labelled confidence level.
- **Chest X-ray:** below 50% pneumonia probability is NORMAL; between 50% and the decision threshold (90%) is **NEEDS REVIEW**; at or above 90% is PNEUMONIA.
- **Brain MRI and symptom checker:** scores under 60% are labelled low confidence and routed to manual review.
- "Not flagged by model" means the model did not flag a finding. It does **not** mean the scan is healthy.
- Grad-CAM heatmaps are an inspection aid, not proof of medical reasoning.

## Measured results (held-out test sets)

| | Brain MRI | Chest X-ray |
|---|---|---|
| Test accuracy | 0.945 | 0.840 (argmax at 0.5) |
| Expected Calibration Error, test | 0.039 | 0.114 |
| Expected Calibration Error, validation | 0.009 | 0.007 |
| Predictions at 90%+ confidence that were wrong | 49 of 1525 | 59 of 533 |

Temperature scaling (fitted on validation only) lowered brain ECE to 0.031 and chest ECE to 0.107, but it is **not applied** in this app.
The gap between validation and test calibration for the chest model is unexplained: it may be dataset shift or a validation split that is too similar to training data.

## Known limitations

- The chest decision threshold (0.90) was chosen on the test set, so chest results are optimistic.
- No out-of-distribution detection: the models will return an answer for any image, including non-medical ones.
- The symptom checker is trained on a small, text-only dataset and has no clinical validation.
- Public datasets only; no external or prospective validation.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...   # optional
python app.py
```

## Repository layout

```
app.py              Gradio app (inference, Grad-CAM, PDF report)
requirements.txt    Pinned dependencies
model/              Trained weights, symptom model, chest threshold config
```

Developed by Aila Nasir.
