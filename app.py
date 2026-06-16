"""
app.py — Streamlit Dashboard for the Federated AI Medical Report Analysis System.

Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from src.utils import (
    preprocess_uploaded_image,
    extract_text_from_image,
    extract_medical_values,
    load_reference_ranges,
    compare_with_reference,
    generate_ai_explanation,
    format_results_dataframe,
    get_status_color,
)
from src.model import (
    train_all_models,
    predict_disease_risk,
    simulate_federated_learning,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REF_RANGES_PATH = os.path.join(_BASE_DIR, "data", "raw", "reference_ranges", "reference_ranges.csv")
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="AI Medical Report Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    /* ---- Global ---- */
    .main { background-color: #0e1117; }

    /* ---- Header banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    .hero-banner p {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ---- Risk gauge ---- */
    .risk-low   { color: #10b981; }
    .risk-med   { color: #f59e0b; }
    .risk-high  { color: #ef4444; }

    /* ---- Status badges ---- */
    .badge-high   { background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-low    { background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-normal { background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

    /* ---- Section headers ---- */
    .section-header {
        border-left: 5px solid #3b82f6;
        padding-left: 15px;
        margin: 2rem 0 1.5rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* ---- Explanation box ---- */
    .explanation-box {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        color: #f1f5f9;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* ---- Federated card ---- */
    .fed-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .fed-card:hover {
        transform: scale(1.02);
    }
    .fed-card h3 { color: #60a5fa; margin-bottom: 0.5rem; font-family: 'Inter', sans-serif; }
    .fed-card .acc { font-size: 2rem; font-weight: 800; color: #10b981; }

    /* Tab styling */
    div[data-testid="stTabs"] button {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    gender = st.selectbox("Patient Gender", ["Male", "Female"], index=0)
    st.markdown("---")

    st.markdown("### 🏥 About")
    st.markdown(
        "This advanced system utilizes **AI & OCR** to process medical reports, "
        "detect anomalous parameters, and leverage state-of-the-art **Machine Learning algorithms** "
        "(XGBoost, Random Forest, LightGBM, SVM, LR) to predict disease risks. "
        "It also incorporates **Federated Learning** to ensure patient privacy across hospital networks."
    )
    st.markdown("---")

    # Train models button
    if st.button("🔄 Train / Optimize Models", use_container_width=True):
        with st.spinner("Training and auto-selecting best algorithms..."):
            metrics = train_all_models()
            st.session_state["model_metrics"] = metrics
        st.success("✅ Models optimized and ready!")

    # Check if models exist
    models_exist = all(
        os.path.exists(os.path.join(_MODELS_DIR, f))
        for f in ["anemia_model.pkl", "ckd_model.pkl",
                   "diabetes_model.pkl", "liver_model.pkl"]
    )
    if models_exist:
        st.markdown("🟢 Models: **Ready (Optimized)**")
    else:
        st.markdown("🔴 Models: **Not trained** — click button above")


# =====================================================================
# HERO BANNER
# =====================================================================
st.markdown("""
<div class="hero-banner">
    <h1>🩺 AI Medical Report Analyzer</h1>
    <p>Upload a blood report → Extract values → Detect anomalies → Predict disease risk with high-accuracy ML models</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 Report Upload & OCR",
    "🔬 Medical Analysis",
    "🩺 Disease Prediction",
    "🤖 AI Explanation",
    "🏥 Federated Learning",
    "📊 Model Performance",
])


# =====================================================================
# TAB 1 — REPORT UPLOAD & OCR
# =====================================================================
with tab1:
    st.markdown('<div class="section-header">Upload Medical Report Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose any report image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="report_uploader",
    )

    if uploaded_file is not None:
        col_img, col_txt = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Report Preview")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col_txt:
            st.markdown("#### 📝 OCR Extracted Text")
            if st.button("🔍 Run Advanced OCR Extraction", key="btn_ocr", use_container_width=True):
                with st.spinner("Running EasyOCR with advanced filtering..."):
                    uploaded_file.seek(0)
                    processed_img = preprocess_uploaded_image(uploaded_file)
                    raw_text = extract_text_from_image(processed_img)
                    st.session_state["ocr_text"] = raw_text
                    st.session_state["extracted_values"] = extract_medical_values(raw_text)

            if "ocr_text" in st.session_state:
                with st.container(height=400):
                    st.code(st.session_state["ocr_text"], language="text")
                st.success(f"**Extracted {len(st.session_state.get('extracted_values', {}))} key medical parameters.**")
    else:
        st.info("👆 Upload a blood report image to get started.")


# =====================================================================
# TAB 2 — MEDICAL ANALYSIS
# =====================================================================
with tab2:
    st.markdown('<div class="section-header">Medical Value Analysis</div>', unsafe_allow_html=True)

    if "extracted_values" in st.session_state and st.session_state["extracted_values"]:
        extracted = st.session_state["extracted_values"]

        # Compare with reference ranges
        ref_df = load_reference_ranges(_REF_RANGES_PATH)
        comparison_df = compare_with_reference(extracted, ref_df, gender)
        formatted_df = format_results_dataframe(comparison_df)
        st.session_state["comparison_df"] = comparison_df

        # Summary counts
        n_high = len(comparison_df[comparison_df["status"] == "HIGH"])
        n_low = len(comparison_df[comparison_df["status"] == "LOW"])
        n_normal = len(comparison_df[comparison_df["status"] == "NORMAL"])

        c1, c2, c3 = st.columns(3)
        with c1:
            if n_high > 0:
                st.error(f"🔴 **{n_high}** parameter(s) are **HIGH**")
            else:
                st.success("No HIGH values detected")
        with c2:
            if n_low > 0:
                st.warning(f"🟠 **{n_low}** parameter(s) are **LOW**")
            else:
                st.success("No LOW values detected")
        with c3:
            st.success(f"🟢 **{n_normal}** parameter(s) are **NORMAL**")

        st.markdown("#### 📊 Reference Range Comparison")
        def highlight_status(row):
            color = get_status_color(row["Status"])
            return [f"color: {color}; font-weight: 700" if col == "Status" else "" for col in row.index]

        styled = formatted_df.style.apply(highlight_status, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    else:
        st.info("⬅️ Go to 'Report Upload & OCR' tab, upload an image, and run OCR first.")


# =====================================================================
# TAB 3 — DISEASE Prediction
# =====================================================================
with tab3:
    st.markdown('<div class="section-header">Disease Risk Prediction</div>', unsafe_allow_html=True)

    if not models_exist:
        st.warning("⚠️ Models not trained yet. Use the sidebar button to optimize and train models first.")
    elif "extracted_values" not in st.session_state or not st.session_state["extracted_values"]:
        st.info("⬅️ Extract medical values from a report first (Tab 1).")
    else:
        extracted = st.session_state["extracted_values"]

        if st.button("🩺 Predict Disease Risks", key="btn_predict", use_container_width=True):
            with st.spinner("Running predictions using optimized ensemble models..."):
                risks = predict_disease_risk(extracted)
                st.session_state["risks"] = risks

        if "risks" in st.session_state:
            risks = st.session_state["risks"]

            cols = st.columns(len(risks))
            for idx, (disease, risk_pct) in enumerate(risks.items()):
                with cols[idx]:
                    if risk_pct < 30:
                        css_class = "risk-low"
                        emoji = "🟢"
                    elif risk_pct < 60:
                        css_class = "risk-med"
                        emoji = "🟡"
                    else:
                        css_class = "risk-high"
                        emoji = "🔴"

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{emoji} {disease}</div>
                        <div class="metric-value {css_class}">{risk_pct}%</div>
                        <div class="metric-label">Risk Score</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Plotly Bar chart for Risk
            st.markdown("#### 📊 Risk Comparison Analysis")
            diseases = list(risks.keys())
            percentages = list(risks.values())
            colors = ["#10b981" if p < 30 else "#f59e0b" if p < 60 else "#ef4444" for p in percentages]

            fig = go.Figure(go.Bar(
                x=percentages,
                y=diseases,
                orientation='h',
                marker_color=colors,
                text=[f"{p}%" for p in percentages],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Inter')
            ))
            fig.update_layout(
                xaxis=dict(title='Risk Percentage (%)', range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title=''),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Inter'),
                margin=dict(l=0, r=0, t=30, b=0),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 4 — AI EXPLANATION
# =====================================================================
with tab4:
    st.markdown('<div class="section-header">AI-Generated Medical Explanations</div>', unsafe_allow_html=True)

    if "comparison_df" in st.session_state:
        explanations = generate_ai_explanation(st.session_state["comparison_df"])

        for exp in explanations:
            st.markdown(f'<div class="explanation-box">{exp}</div>', unsafe_allow_html=True)

        if "risks" in st.session_state:
            st.markdown("---")
            st.markdown("#### 🩺 Comprehensive Risk Summary")
            for disease, risk in st.session_state["risks"].items():
                if risk >= 60:
                    st.error(f"🔴 **{disease}**: {risk}% risk — **High clinical risk detected.** Immediate medical evaluation is strongly recommended.")
                elif risk >= 30:
                    st.warning(f"🟡 **{disease}**: {risk}% risk — **Moderate risk.** Regular monitoring and follow-up consultation are advised.")
                else:
                    st.success(f"🟢 **{disease}**: {risk}% risk — **Low risk.** Parameters are within optimal ranges.")
    else:
        st.info("⬅️ Run medical analysis first (Tab 1 → Tab 2).")


# =====================================================================
# TAB 5 — FEDERATED LEARNING
# =====================================================================
with tab5:
    st.markdown('<div class="section-header">Federated Learning Simulation</div>', unsafe_allow_html=True)

    st.markdown("""
    **Privacy-Preserving Architecture**: Patient data remains on local hospital servers. 
    Only model weights are aggregated to create a robust global model.
    """)

    fl_col1, fl_col2 = st.columns([1, 1])
    with fl_col1:
        fl_dataset = st.selectbox("Select Disease Dataset", ["diabetes", "anemia", "ckd", "liver"])
    with fl_col2:
        fl_hospitals = st.slider("Number of Distributed Nodes (Hospitals)", 2, 8, 3)

    if st.button("🚀 Run Federated Simulation", key="btn_fl", use_container_width=True):
        with st.spinner("Simulating distributed training and aggregation..."):
            fl_results = simulate_federated_learning(fl_dataset, fl_hospitals)
            st.session_state["fl_results"] = fl_results

    if "fl_results" in st.session_state:
        fl = st.session_state["fl_results"]

        # Hospital cards
        cols = st.columns(len(fl["hospital_results"]))
        for idx, res in enumerate(fl["hospital_results"]):
            with cols[idx]:
                st.markdown(f"""
                <div class="fed-card">
                    <h3>🏥 {res['hospital']}</h3>
                    <div style="color:#94a3b8; font-size:0.95rem;">{res['samples']} samples</div>
                    <div class="acc">{res['accuracy']}%</div>
                    <div style="color:#94a3b8; font-size:0.9rem;">Local Accuracy</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius:12px; padding:1.5rem; margin-top:1.5rem; text-align:center;">
            <strong style="color:#10b981; font-size: 1.5rem;">✅ Aggregated Global Accuracy: {fl['average_accuracy']}%</strong><br>
            <span style="color:#cbd5e1; font-size: 1.1rem;">Privacy Preserved — Zero raw patient data was transmitted across the network.</span>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Bar Chart
        st.markdown("<br>#### 📊 Node Accuracy Distribution", unsafe_allow_html=True)
        hospitals = [r["hospital"] for r in fl["hospital_results"]]
        accuracies = [r["accuracy"] for r in fl["hospital_results"]]

        fig = go.Figure(go.Bar(
            x=hospitals,
            y=accuracies,
            marker_color='#3b82f6',
            text=[f"{a}%" for a in accuracies],
            textposition='auto',
            textfont=dict(color='white', size=14, family='Inter'),
            marker_line_width=0
        ))
        fig.update_layout(
            xaxis=dict(title=''),
            yaxis=dict(title='Accuracy (%)', range=[max(0, min(accuracies)-10), 105], gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 6 — MODEL PERFORMANCE
# =====================================================================
with tab6:
    st.markdown('<div class="section-header">Optimized Model Metrics</div>', unsafe_allow_html=True)

    if "model_metrics" not in st.session_state:
        if models_exist:
            if st.button("📊 Load Metrics", key="btn_metrics", use_container_width=True):
                with st.spinner("Evaluating optimized models..."):
                    metrics = train_all_models()
                    st.session_state["model_metrics"] = metrics
        else:
            st.info("Optimize models first using the sidebar button.")

    if "model_metrics" in st.session_state:
        metrics = st.session_state["model_metrics"]

        # Summary table
        summary_rows = []
        for disease, m in metrics.items():
            summary_rows.append({
                "Disease": disease,
                "Best Algorithm": m.get("best_algorithm", "Unknown"),
                "Accuracy (%)": m["accuracy"],
                "Precision (%)": m["precision"],
                "Recall (%)": m["recall"],
                "F1 Score (%)": m["f1_score"],
            })
        summary_df = pd.DataFrame(summary_rows)

        st.markdown("#### 📋 Performance Overview")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Plotly grouped bar chart
        st.markdown("#### 📊 Accuracy & F1-Score Comparison")
        diseases = list(metrics.keys())
        accs = [metrics[d]["accuracy"] for d in diseases]
        f1s = [metrics[d]["f1_score"] for d in diseases]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=diseases, y=accs, name='Accuracy', marker_color='#3b82f6',
            text=[f"{a}%" for a in accs], textposition='auto'
        ))
        fig.add_trace(go.Bar(
            x=diseases, y=f1s, name='F1 Score', marker_color='#8b5cf6',
            text=[f"{f}%" for f in f1s], textposition='auto'
        ))

        fig.update_layout(
            barmode='group',
            xaxis=dict(title=''),
            yaxis=dict(title='Score (%)', range=[0, 110], gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrices using Plotly
        st.markdown("#### 🧮 Confusion Matrices")
        cm_cols = st.columns(2, gap="large")
        for idx, (disease, m) in enumerate(metrics.items()):
            with cm_cols[idx % 2]:
                st.markdown(f"**{disease}** ({m.get('best_algorithm', '')})")
                cm = np.array(m["confusion_matrix"])
                
                # Flip the matrix for plotly so it displays like seaborn (origin top-left)
                cm_text = [[str(val) for val in row] for row in cm]
                
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Negative', 'Positive'], y=['Negative', 'Positive'])
                fig_cm.update_layout(
                    coloraxis_showscale=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=250
                )
                st.plotly_chart(fig_cm, use_container_width=True)


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#64748b; font-size:0.9rem; font-family: \'Inter\', sans-serif;">'
    '🩺 <strong>Federated AI Medical Report Analyzer</strong> &nbsp;|&nbsp; '
    'Professional MCA Level Project &nbsp;|&nbsp; '
    'Powered by Streamlit, XGBoost & Plotly'
    '</div>',
    unsafe_allow_html=True,
)
