# multilingual-ocr-translation

A deep learning pipeline for Hindi and Bengali text detection, language identification, and English translation — built on a custom-trained CRNN model with confidence-aware decision making.

## What It Does

Given any image containing Hindi or Bengali text, the pipeline:
1. Detects all text regions in the image
2. Runs both Hindi and Bengali OCR readers on each region
3. Selects the best result using confidence scoring + Unicode script detection
4. For low-confidence regions — invokes a CRNN-based character recognizer to validate the script, then retries OCR with the recommended reader
5. Translates high-confidence results to English
6. Visualizes all bounding boxes color-coded by language and confidence

## Pipeline Architecture

```
Image
  └── EasyOCR (English detector) → text regions
        └── For each region:
              ├── reader_hi (Hindi + English)
              ├── reader_bn (Bengali + English)
              └── Confidence scoring
                    ├── conf >= 0.50 → Translate directly
                    └── conf < 0.50  → CRNN script validator
                                          └── Split crop into character patches
                                              └── CRNN majority vote → hi / bn
                                                    └── Retry OCR with recommended reader
                                                          ├── conf >= 0.50 → Translate
                                                          └── still low   → Skip (red box)
```

## Confidence-Aware Decision Making

Per-region multilingual OCR is achieved by evaluating multiple language models and selecting outputs using a combination of:
- **Confidence scoring** — higher confidence reader wins
- **Unicode range validation** — Devanagari (`0x0900–0x097F`) vs Bengali (`0x0980–0x09FF`) used as tiebreaker when confidences are close
- **CRNN script validation** — for low-confidence regions, a trained CRNN character recognizer votes on the script at patch level, preventing bad OCR from being passed to the translator

For low-confidence OCR outputs, translation is skipped entirely to prevent error propagation.

## Tech Stack

- EasyOCR — text detection and recognition
- PyTorch — CRNN training
- Helsinki-NLP/opus-mt-hi-en — Hindi → English translation
- Helsinki-NLP/opus-mt-bn-en — Bengali → English translation
- OpenCV + Matplotlib — bounding box visualization
- ICDAR 2019 MLT Dataset — real-world multilingual scene text

## Datasets

| Dataset | Images | Used For |
|---|---|---|
| Devanagari Character Dataset (nhcd) | 12.9k | CRNN training — Hindi characters |
| BanglaLekha-Isolated | 166k | CRNN training — Bengali characters |
| ICDAR 2019 MLT | — | Pipeline testing — real scene images |
| Custom Hindi demo (tistii) | 4 images | Professor demo |

## CRNN Training Results

| Epoch | Loss | Accuracy |
|---|---|---|
| 1 | 3.91 | 7.12% |
| 5 | 1.57 | 57.11% |
| 10 | 0.32 | 93.70% |

- Trained from scratch on 6,050 samples across 94 character classes
- Input: 32×128 grayscale character patches
- Architecture: 3× Conv-ReLU-MaxPool → BiLSTM → FC
- Trained on Kaggle GPU T4

## Bounding Box Color Legend

| Color | Meaning |
|---|---|
| 🟢 Green | Hindi — high confidence — translated |
| 🔵 Blue | Bengali — high confidence — translated |
| 🟡 Yellow | Hindi — recovered by CRNN — translated |
| 🟠 Orange | Bengali — recovered by CRNN — translated |
| 🔴 Red | Low confidence — CRNN used, still failed — translation skipped |

## Notebooks

- `final_pipeline.ipynb` — single notebook: CRNN training + full OCR pipeline in one session

## Known Limitations

- Hindi and Marathi share Devanagari script and are indistinguishable at character level
- Highly stylized or decorative fonts produce lower OCR confidence
- Tamil was dropped from scope due to broken model weights in Kaggle EasyOCR version

## Environment

Kaggle Notebooks — GPU T4 — Python 3.12
