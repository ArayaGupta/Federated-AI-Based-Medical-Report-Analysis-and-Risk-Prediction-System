"""
utils.py — Utility functions for the Federated AI Medical Report Analysis System.

Provides:
  - Image preprocessing (OpenCV)
  - OCR text extraction (EasyOCR)
  - Regex-based medical value extraction
  - Reference range loading and comparison
  - AI-based medical explanation generation
  - Formatting utilities
"""

import re
import os
import sys
import io
import cv2
import numpy as np
import pandas as pd

# Fix Windows console encoding issues with EasyOCR's progress bar
os.environ["PYTHONIOENCODING"] = "utf-8"

import easyocr

# ---------------------------------------------------------------------------
# Lazy-initialized global OCR reader (expensive to create)
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_ocr_reader():
    """Return a cached EasyOCR reader instance (English)."""
    global _ocr_reader
    if _ocr_reader is None:
        # verbose=False suppresses the download progress bar that causes
        # UnicodeEncodeError on Windows with cp1252 encoding
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


# =====================================================================
# 1. IMAGE PREPROCESSING
# =====================================================================

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from *image_path* and apply preprocessing to
    improve OCR accuracy.

    Steps:
        1. Read image with OpenCV
        2. Convert to grayscale
        3. Apply CLAHE (adaptive histogram equalization)
        4. Denoise with Non-Local Means
        5. Apply adaptive thresholding for binarization

    Returns the preprocessed image as a NumPy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE – improves contrast on scanned documents
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    return denoised


def preprocess_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Accept a Streamlit UploadedFile (or any file-like bytes buffer)
    and return a preprocessed NumPy image ready for OCR.
    """
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return denoised


# =====================================================================
# 2. OCR TEXT EXTRACTION
# =====================================================================

def extract_text_from_image(image) -> str:
    """
    Run EasyOCR on a preprocessed image (numpy array) or a file path
    and return the merged text output as a single string.
    """
    reader = _get_ocr_reader()

    if isinstance(image, str):
        # It's a file path – preprocess first
        image = preprocess_image(image)

    results = reader.readtext(image, detail=0, paragraph=False)
    merged_text = "\n".join(results)
    return merged_text


def extract_text_from_path(image_path: str) -> str:
    """Convenience wrapper: path → preprocessed image → OCR text."""
    processed = preprocess_image(image_path)
    return extract_text_from_image(processed)


# =====================================================================
# 3. MEDICAL VALUE EXTRACTION (REGEX)
# =====================================================================

# Each entry: (display_name, regex_pattern, list_of_aliases)
_MEDICAL_PATTERNS = [
    # Hematology
    ("Hemoglobin",      r"(?:Haemoglobin|Hemoglobin|Hb|HGB)\s*[:\-]?\s*(\d+\.?\d*)",      ["hemoglobin", "hb", "hgb"]),
    ("WBC",             r"(?:WBC|White\s*Blood\s*Cell|Total\s*WBC|Leucocyte)\s*[:\-]?\s*(\d+\.?\d*)", ["wbc"]),
    ("RBC",             r"(?:RBC|Red\s*Blood\s*Cell|Erythrocyte)\s*[:\-]?\s*(\d+\.?\d*)",   ["rbc"]),
    ("Platelets",       r"(?:Platelet|PLT)\s*(?:Count)?\s*[:\-]?\s*(\d+\.?\d*)",            ["platelets", "plt"]),
    ("MCV",             r"(?:MCV|Mean\s*Corpuscular\s*Volume)\s*[:\-]?\s*(\d+\.?\d*)",       ["mcv"]),
    ("MCH",             r"(?:MCH|Mean\s*Corpuscular\s*Hemo)\s*[:\-]?\s*(\d+\.?\d*)",        ["mch"]),
    ("MCHC",            r"(?:MCHC|Mean\s*Corpuscular\s*Hemo\s*Conc)\s*[:\-]?\s*(\d+\.?\d*)",["mchc"]),

    # Renal
    ("Serum Creatinine",r"(?:Creatinine|Serum\s*Creatinine|S\.\s*Creatinine|Sr\.\s*Creatinine)\s*[:\-]?\s*(\d+\.?\d*)", ["creatinine"]),
    ("Blood Urea",      r"(?:Blood\s*Urea|Urea|BUN)\s*[:\-]?\s*(\d+\.?\d*)",               ["urea", "bun"]),
    ("Sodium",          r"(?:Sodium|Na\+?|Serum\s*Sodium)\s*[:\-]?\s*(\d+\.?\d*)",          ["sodium", "na"]),
    ("Potassium",       r"(?:Potassium|K\+?|Serum\s*Potassium)\s*[:\-]?\s*(\d+\.?\d*)",     ["potassium", "k"]),

    # Metabolic
    ("Fasting Glucose", r"(?:Fasting\s*(?:Blood\s*)?Glucose|FBS|Glucose\s*Fasting|Blood\s*(?:Sugar|Glucose))\s*[:\-]?\s*(\d+\.?\d*)", ["glucose", "fbs"]),
    ("HbA1c",           r"(?:HbA1c|Glycated\s*Hemoglobin|A1C)\s*[:\-]?\s*(\d+\.?\d*)",     ["hba1c", "a1c"]),

    # Lipids
    ("Total Cholesterol",r"(?:Total\s*Cholesterol|Cholesterol)\s*[:\-]?\s*(\d+\.?\d*)",     ["cholesterol"]),
    ("LDL",             r"(?:LDL|Low\s*Density)\s*[:\-]?\s*(\d+\.?\d*)",                    ["ldl"]),
    ("HDL",             r"(?:HDL|High\s*Density)\s*[:\-]?\s*(\d+\.?\d*)",                   ["hdl"]),
    ("Triglycerides",   r"(?:Triglycerides|TG)\s*[:\-]?\s*(\d+\.?\d*)",                     ["triglycerides", "tg"]),

    # Liver
    ("Total Bilirubin", r"(?:Total\s*Bilirubin|T[\.\s]*Bilirubin|Bilirubin\s*Total)\s*[:\-]?\s*(\d+\.?\d*)", ["bilirubin"]),
    ("Direct Bilirubin",r"(?:Direct\s*Bilirubin|D[\.\s]*Bilirubin|Conjugated\s*Bilirubin)\s*[:\-]?\s*(\d+\.?\d*)", ["direct_bilirubin"]),
    ("SGPT",            r"(?:SGPT|ALT|Alamine\s*Aminotransferase)\s*[:\-]?\s*(\d+\.?\d*)",  ["sgpt", "alt"]),
    ("SGOT",            r"(?:SGOT|AST|Aspartate\s*Aminotransferase)\s*[:\-]?\s*(\d+\.?\d*)",["sgot", "ast"]),
    ("Alkaline Phosphatase", r"(?:Alkaline\s*Phosph[oa]tase|ALP)\s*[:\-]?\s*(\d+\.?\d*)",   ["alp"]),
    ("Total Proteins",  r"(?:Total\s*Prote[ie]ns?|TP)\s*[:\-]?\s*(\d+\.?\d*)",              ["total_proteins", "tp"]),
    ("Albumin",         r"(?:Albumin|Serum\s*Albumin)\s*[:\-]?\s*(\d+\.?\d*)",              ["albumin"]),
]


def extract_medical_values(text: str) -> dict:
    """
    Parse OCR text and extract medical test values using regex.
    Includes text normalization to handle common OCR artifacts.
    """
    # Normalize text: fix common OCR misreads
    text = text.replace("O", "0").replace("l", "1").replace(",", ".")
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces

    extracted = {}
    for display_name, pattern, _aliases in _MEDICAL_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                extracted[display_name] = float(match.group(1))
            except ValueError:
                continue
    return extracted


# =====================================================================
# 4. REFERENCE RANGE LOADING & COMPARISON
# =====================================================================

def load_reference_ranges(csv_path: str) -> pd.DataFrame:
    """
    Load the reference_ranges.csv file and return a DataFrame
    with columns: test_name, unit, gender, age_group, ref_low, ref_high.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def compare_with_reference(
    extracted_values: dict,
    ref_df: pd.DataFrame,
    gender: str = "All",
) -> pd.DataFrame:
    """
    Compare extracted medical values against reference ranges.

    Parameters
    ----------
    extracted_values : dict
        Mapping of test name → numeric value.
    ref_df : DataFrame
        Reference ranges loaded via ``load_reference_ranges``.
    gender : str
        'Male', 'Female', or 'All'.

    Returns
    -------
    DataFrame with columns: test, value, unit, ref_low, ref_high, status
    """
    rows = []
    for test_name, value in extracted_values.items():
        # Try gender-specific match first, then 'All'
        mask = ref_df["test_name"].str.lower() == test_name.lower()
        matched = ref_df[mask]

        if matched.empty:
            # Fallback: partial match
            mask_partial = ref_df["test_name"].str.lower().str.contains(
                test_name.lower().split()[0], na=False
            )
            matched = ref_df[mask_partial]

        if matched.empty:
            rows.append({
                "test": test_name, "value": value,
                "unit": "-", "ref_low": "-", "ref_high": "-",
                "status": "NO REF",
            })
            continue

        # Prefer gender-specific row
        gender_match = matched[matched["gender"].str.lower() == gender.lower()]
        if gender_match.empty:
            gender_match = matched[matched["gender"].str.lower() == "all"]
        if gender_match.empty:
            gender_match = matched.head(1)

        row = gender_match.iloc[0]
        ref_low = float(row["ref_low"])
        ref_high = float(row["ref_high"])
        unit = row.get("unit", "-")

        if value < ref_low:
            status = "LOW"
        elif value > ref_high:
            status = "HIGH"
        else:
            status = "NORMAL"

        rows.append({
            "test": test_name, "value": value,
            "unit": unit, "ref_low": ref_low, "ref_high": ref_high,
            "status": status,
        })

    return pd.DataFrame(rows)


# =====================================================================
# 5. AI MEDICAL EXPLANATION GENERATOR
# =====================================================================

_EXPLANATION_MAP = {
    "Hemoglobin": {
        "LOW": "Low Hemoglobin indicates potential anemia, blood loss, or nutritional deficiency (iron, B12, folate). Clinical correlation is advised.",
        "HIGH": "Elevated Hemoglobin may be associated with dehydration, polycythemia vera, or chronic hypoxia.",
    },
    "WBC": {
        "LOW": "Leukopenia (Low WBC) may indicate bone marrow suppression, severe viral infections, or autoimmune conditions.",
        "HIGH": "Leukocytosis (High WBC) is typically a marker of systemic infection, severe inflammation, or hematologic malignancies.",
    },
    "RBC": {
        "LOW": "Decreased RBC count is indicative of anemia, active hemorrhage, or bone marrow dysfunction.",
        "HIGH": "Elevated RBC may indicate polycythemia, chronic pulmonary disease, or profound dehydration.",
    },
    "Platelets": {
        "LOW": "Thrombocytopenia (Low Platelets) raises the risk of bleeding and may indicate viral infections (e.g., dengue), or autoimmune destruction.",
        "HIGH": "Thrombocytosis (High Platelets) is commonly seen in acute infection, chronic inflammation, or myeloproliferative neoplasms.",
    },
    "MCV": {
        "LOW": "Microcytosis (Low MCV) points toward iron deficiency anemia or thalassemia traits.",
        "HIGH": "Macrocytosis (High MCV) is primarily caused by Vitamin B12 or folate deficiency.",
    },
    "MCH": {
        "LOW": "Low MCH frequently correlates with hypochromic microcytic anemia, most commonly iron deficiency.",
        "HIGH": "High MCH is associated with macrocytic anemias.",
    },
    "MCHC": {
        "LOW": "Decreased MCHC signifies hypochromia, often reflecting iron deficiency.",
        "HIGH": "Elevated MCHC may indicate hereditary spherocytosis or autoimmune hemolytic anemia.",
    },
    "Serum Creatinine": {
        "LOW": "Low creatinine is usually benign but can reflect decreased muscle mass.",
        "HIGH": "Elevated Creatinine is a primary marker for impaired renal function or Chronic Kidney Disease (CKD). Nephrology evaluation is indicated.",
    },
    "Blood Urea": {
        "LOW": "Low urea may be seen in severe hepatic impairment or protein malnutrition.",
        "HIGH": "Elevated urea (azotemia) is indicative of renal insufficiency, significant dehydration, or upper GI bleeding.",
    },
    "Sodium": {
        "LOW": "Hyponatremia may be due to renal sodium loss, SIADH, or excessive fluid intake.",
        "HIGH": "Hypernatremia indicates a free water deficit or severe dehydration.",
    },
    "Potassium": {
        "LOW": "Hypokalemia can lead to significant muscle weakness and cardiac arrhythmias.",
        "HIGH": "Hyperkalemia is a medical emergency that can precipitate fatal cardiac arrhythmias, often seen in acute or chronic renal failure.",
    },
    "Fasting Glucose": {
        "LOW": "Hypoglycemia may cause autonomic and neuroglycopenic symptoms (dizziness, confusion).",
        "HIGH": "Hyperglycemia in a fasting state strongly suggests Impaired Fasting Glucose (IFG) or Diabetes Mellitus.",
    },
    "HbA1c": {
        "LOW": "Low HbA1c indicates excellent glycemic control, or rarely, hemolytic conditions.",
        "HIGH": "Elevated HbA1c (>6.5%) is diagnostic of Diabetes Mellitus, indicating poor long-term glycemic control.",
    },
    "Total Cholesterol": {
        "HIGH": "Hypercholesterolemia significantly increases the risk of atherogenesis and cardiovascular disease.",
    },
    "LDL": {
        "HIGH": "Elevated LDL ('bad' cholesterol) is a primary risk factor for atherosclerosis, coronary artery disease, and stroke.",
    },
    "HDL": {
        "LOW": "Low HDL ('good' cholesterol) is an independent risk factor for cardiovascular disease.",
    },
    "Triglycerides": {
        "HIGH": "Hypertriglyceridemia increases cardiovascular risk and, if severe, the risk of acute pancreatitis.",
    },
    "Total Bilirubin": {
        "HIGH": "Hyperbilirubinemia may indicate hepatocellular damage, biliary obstruction, or increased hemolysis.",
    },
    "Direct Bilirubin": {
        "HIGH": "Elevated conjugated (direct) bilirubin typically indicates cholestasis, biliary obstruction, or hepatocellular injury.",
    },
    "SGPT": {
        "HIGH": "Elevated SGPT (ALT) is highly specific for hepatocellular injury, such as viral hepatitis or non-alcoholic fatty liver disease (NAFLD).",
    },
    "SGOT": {
        "HIGH": "Elevated SGOT (AST) indicates hepatocellular damage, but may also rise in myocardial infarction or skeletal muscle injury.",
    },
    "Alkaline Phosphatase": {
        "HIGH": "Elevated ALP indicates biliary tract obstruction (cholestasis) or increased bone turnover.",
    },
    "Total Proteins": {
        "LOW": "Hypoproteinemia may indicate malnutrition, severe hepatic dysfunction, or protein-losing enteropathy/nephropathy.",
        "HIGH": "Hyperproteinemia may indicate chronic inflammatory states, dehydration, or paraproteinemias (e.g., multiple myeloma).",
    },
    "Albumin": {
        "LOW": "Hypoalbuminemia is a marker of diminished hepatic synthesis, systemic inflammation, or significant renal/gastrointestinal loss.",
    },
}


def generate_ai_explanation(comparison_df: pd.DataFrame) -> list:
    """
    Generate human-readable AI explanations for abnormal findings.

    Parameters
    ----------
    comparison_df : DataFrame
        Output of ``compare_with_reference`` with a 'status' column.

    Returns
    -------
    List of explanation strings.
    """
    explanations = []
    abnormal = comparison_df[comparison_df["status"].isin(["HIGH", "LOW"])]

    for _, row in abnormal.iterrows():
        test = row["test"]
        status = row["status"]
        value = row["value"]

        explanation_dict = _EXPLANATION_MAP.get(test, {})
        detail = explanation_dict.get(status, f"{test} is {status} which may require clinical evaluation.")

        explanations.append(
            f"⚠️ **{test}** = {value} → **{status}**\n   {detail}"
        )

    if not explanations:
        explanations.append("✅ All extracted parameters are within normal reference ranges.")

    return explanations


# =====================================================================
# 6. FORMATTING UTILITIES
# =====================================================================

def format_results_dataframe(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a presentation-ready copy of the comparison DataFrame
    with clean column names and sorted by status (abnormal first).
    """
    df = comparison_df.copy()
    df.columns = ["Test", "Value", "Unit", "Ref Low", "Ref High", "Status"]
    status_order = {"HIGH": 0, "LOW": 1, "NORMAL": 2, "NO REF": 3}
    df["_sort"] = df["Status"].map(status_order).fillna(4)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def get_status_color(status: str) -> str:
    """Return a CSS-friendly color string for a given status."""
    return {
        "HIGH": "#e74c3c",
        "LOW": "#e67e22",
        "NORMAL": "#2ecc71",
        "NO REF": "#95a5a6",
    }.get(status, "#bdc3c7")
