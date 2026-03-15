# multilingual-ocr-translation
A deep learning pipeline for Hindi and Bengali text detection, language identification, and English translation using fine-tuned CRNN model.

A Python pipeline that detects Hindi and Bengali text in images, identifies the language, and translates to English.

## Tech Stack
- EasyOCR (CRNN model)
- PyTorch (fine-tuning)
- Unicode-based script detection
- deep-translator (Google Translate)
- ICDAR 2019 MLT Dataset

## Dataset
- Devanagari Character Dataset (12.9k images)
- BanglaLekha-Isolated (166k images)

## Results
- Fine-tuned CRNN: 93.7% accuracy after 10 epochs
- Loss reduced from 3.91 → 0.32

## Notebooks
- `finetuning.ipynb` - CRNN fine-tuning on Hindi/Bengali data
- `pipeline.ipynb` - End-to-end detection + translation pipeline

## Environment
Kaggle Notebooks (GPU T4)
