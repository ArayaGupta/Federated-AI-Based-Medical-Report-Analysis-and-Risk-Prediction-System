# Federated AI-Based Medical Report Analysis and Disease Prediction System

An AI-powered healthcare system that analyzes blood report images, detects abnormalities, predicts disease risks, and demonstrates federated learning for privacy-preserving healthcare AI.

## Features

- **OCR-based Report Analysis** — EasyOCR extracts text from scanned blood reports
- **Medical Value Extraction** — Regex-based extraction of 20+ medical parameters
- **Abnormality Detection** — Compares values against standard reference ranges
- **Disease Prediction** — Random Forest models predict risk for Anemia, CKD, Diabetes, and Liver Disease
- **AI Explanations** — Rule-based medical explanations for abnormal findings
- **Federated Learning** — Privacy-preserving simulation across virtual hospitals
- **Streamlit Dashboard** — Interactive web interface with visualizations

## Project Structure

```
REPORT/
├── data/raw/
│   ├── lab_reports/         # Scanned blood report images (426 files)
│   ├── reference_ranges/    # reference_ranges.csv
│   └── risk_datasets/       # anemia, ckd, diabetes, liver CSVs
├── models/                  # Saved .pkl model files
├── src/
│   ├── model.py             # ML training, prediction, federated learning
│   └── utils.py             # OCR, regex, reference comparison, AI explanations
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models
```bash
python -m src.model
```

### 2. Launch Dashboard
```bash
python -m streamlit run app.py
```

### 3. Workflow
1. Upload a blood report image
2. Click "Run OCR Extraction"
3. View extracted values and abnormalities in the Medical Analysis tab
4. Check disease risk predictions
5. Read AI-generated explanations
6. Explore federated learning simulation

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming |
| EasyOCR | OCR text extraction |
| Pandas / NumPy | Data processing |
| Scikit-learn | Random Forest ML models |
| OpenCV | Image preprocessing |
| Streamlit | Web dashboard |
| Matplotlib / Seaborn | Visualization |
| Joblib | Model serialization |

## Disease Models

| Model | Dataset | Target | Accuracy |
|---|---|---|---|
| Anemia | anemia.csv | Result | ~95%+ |
| CKD | ckd.csv | Class | ~97%+ |
| Diabetes | diabetes_prediction.csv | diabetes | ~96%+ |
| Liver Disease | liver.csv | Dataset | ~72%+ |

## Federated Learning

The system simulates federated learning by:
1. Splitting datasets into N subsets (representing hospitals)
2. Training independent Random Forest models on each subset
3. Comparing per-hospital accuracies
4. Demonstrating that privacy is preserved — no raw data shared

## Future Scope

- ECG/X-ray integration
- LLM-based explanations (GPT/Gemini)
- Real hospital deployment
- Cloud integration (AWS/GCP)
- Multilingual OCR support
- Real federated averaging (FedAvg)
