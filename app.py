"""
Streamlit Web Application for Cyberbullying & Toxic Comment Detection.
Provides interactive live prediction, toxicity gauge, architecture inspector, and evaluation visualizer.
Supports both modular subfolder structure and flat GitHub root structure.
"""

import os
import sys
import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Path handling: support both subfolder and root repository layouts
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.append(src_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from predict import CyberbullyingPredictor

# Page Configuration
st.set_page_config(
    page_title="CyberGuard DL - Cyberbullying & Toxic Content Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #EF4444;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-toxic {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper to find file in subfolder or root
def get_file_path(subfolder: str, filename: str) -> str:
    path_sub = os.path.join(current_dir, subfolder, filename)
    if os.path.exists(path_sub):
        return path_sub
    path_root = os.path.join(current_dir, filename)
    if os.path.exists(path_root):
        return path_root
    return path_sub

# Cache model predictor
@st.cache_resource
def load_predictor():
    model_path = get_file_path("saved_models", "bilstm_model.keras")
    tokenizer_path = get_file_path("saved_models", "tokenizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None
    return CyberbullyingPredictor(model_path, tokenizer_path)

predictor = load_predictor()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=70)
st.sidebar.title("🛡️ CyberGuard DL")
st.sidebar.markdown("**CA1 Deep Learning Project**")
st.sidebar.markdown("*NLP & Social Media Safety*")
st.sidebar.divider()

st.sidebar.markdown("### 📌 Architecture Overview")
st.sidebar.markdown("""
- **Model**: Bidirectional LSTM (BiLSTM)
- **Framework**: TensorFlow / Keras 3
- **Embeddings**: 128-dim Learned Vectors
- **Sequence Length**: 100 Tokens
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.001)
""")

st.sidebar.divider()
st.sidebar.info("💡 **Viva Tip**: Bidirectional LSTM preserves context from both preceding and succeeding tokens, crucial for sarcasm and context-dependent insults.")

# Header
st.markdown('<div class="main-header">🛡️ CyberGuard DL: Toxic Comment & Cyberbullying Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning-powered real-time NLP classification using Bidirectional Recurrent Neural Networks (TensorFlow/Keras)</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Comment Scanner",
    "🧠 Model Architecture & Theory",
    "📊 Evaluation & Metrics",
    "📁 Dataset Explorer"
])

# ----------------- TAB 1: Real-Time Scanner -----------------
with tab1:
    st.subheader("Interactive Social Media Comment Analysis")
    st.write("Type or paste any comment below to analyze its toxicity and cyberbullying probability.")
    
    # Preset Examples
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    preset_text = ""
    with col_pre1:
        if st.button("📝 Safe Educational Comment"):
            preset_text = "Thank you so much for the helpful deep learning tutorial! It really clarified backpropagation for me."
    with col_pre2:
        if st.button("💬 Polite Disagreement"):
            preset_text = "I respect your opinion, but I disagree with this approach because the computational complexity is too high."
    with col_pre3:
        if st.button("⚠️ Hostile Insult"):
            preset_text = "You are completely stupid and incompetent, nobody wants to hear your useless opinion."
    with col_pre4:
        if st.button("🚨 Severe Harassment"):
            preset_text = "You are a disgusting freak and a waste of space, go jump off a cliff nobody cares about you."
            
    input_text = st.text_area(
        "Enter Comment / Social Media Text:",
        value=preset_text if preset_text else "Great presentation today! Really enjoyed the clear diagrams.",
        height=110,
        placeholder="Type text here..."
    )
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
        
    if analyze_btn or input_text:
        if predictor is None:
            st.warning("⚠️ Model weights not found! Please check that `bilstm_model.keras` and `tokenizer.pkl` are present.")
        else:
            with st.spinner("Analyzing text via Bidirectional LSTM..."):
                res = predictor.predict(input_text)
                
            st.divider()
            
            # Result Banner
            col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 2])
            
            with col_res1:
                st.markdown("### Detection Verdict")
                if res['is_toxic']:
                    st.markdown(f'<div class="badge-toxic">⚠️ {res["label"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-safe">✅ {res["label"]}</div>', unsafe_allow_html=True)
                st.caption(f"Risk Rating: **{res['severity']}**")
                
            with col_res2:
                st.markdown("### Toxicity Probability")
                st.metric(label="Calculated DL Score", value=f"{res['confidence_pct']}%")
                
            with col_res3:
                st.markdown("### Threat Hazard Level")
                # Custom color progress bar
                st.progress(res['probability'])
                if res['probability'] < 0.25:
                    st.caption("🟢 Low Risk: Content conforms to community guidelines.")
                elif res['probability'] < 0.50:
                    st.caption("🟡 Moderate: Benign or casual conversation.")
                elif res['probability'] < 0.80:
                    st.caption("🟠 High Risk: Hostile or degrading content detected.")
                else:
                    st.caption("🔴 Critical: Severe cyberbullying / direct harassment violation.")
                    
            # Details Section
            st.markdown("#### Linguistic Analysis & Flagged Terms")
            col_det1, col_det2 = st.columns(2)
            with col_det1:
                st.write("**Preprocessed Token Stream:**")
                st.code(res['cleaned_text'] if res['cleaned_text'] else "[Empty Sequence]")
            with col_det2:
                st.write("**Identified Flagged Hostile Keywords:**")
                if res['flagged_keywords']:
                    st.error(", ".join(res['flagged_keywords']))
                else:
                    st.success("No explicit aggressive keywords matched. Classification driven by deep contextual sequence embeddings.")

# ----------------- TAB 2: Architecture & Theory -----------------
with tab2:
    st.subheader("🧠 Deep Learning Architecture & Mathematical Formulation")
    
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("### 1. Bidirectional LSTM Pipeline")
        st.markdown("""
        Standard Recurrent Neural Networks (RNNs) suffer from **vanishing and exploding gradients** when processing longer text sequences due to continuous matrix multiplications:
        $$\\frac{\\partial L}{\\partial W} = \\sum_{t=1}^{T} \\frac{\\partial L_t}{\\partial h_t} \\prod_{j=k+1}^{t} \\frac{\\partial h_j}{\\partial h_{j-1}} \\frac{\\partial h_k}{\\partial W}$$
        
        **Long Short-Term Memory (LSTM)** addresses this through gating mechanisms:
        - **Forget Gate ($f_t$)**: Decides what past information to discard:
          $$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$$
        - **Input Gate ($i_t$) & Candidate State ($\\tilde{C}_t$)**: Determines which new signals to store:
          $$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$$
          $$\\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)$$
        - **Cell State ($C_t$)**: Additive update preventing vanishing gradients:
          $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$
        - **Output Gate ($o_t$) & Hidden State ($h_t$)**:
          $$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$$
          $$h_t = o_t \\odot \\tanh(C_t)$$
          
        **Bidirectional Processing**:
        $$\\vec{h}_t = \\text{LSTM}_{\\text{forward}}(x_t, \\vec{h}_{t-1})$$
        $$\\overleftarrow{h}_t = \\text{LSTM}_{\\text{backward}}(x_t, \\overleftarrow{h}_{t+1})$$
        $$h_t^{\\text{concat}} = [\\vec{h}_t \\,;\\, \\overleftarrow{h}_t]$$
        """)
        
    with col_arch2:
        st.markdown("### 2. Layer-by-Layer Specifications")
        st.markdown("""
| Layer | Specification | Parameter / Operation |
|---|---|---|
| **Input** | `(Batch, 100)` | Fixed token length vector |
| **Embedding** | `(10000, 128)` | Learned 128-D dense semantic space |
| **SpatialDropout1D** | `Rate = 0.30` | Drops entire 1D word feature maps |
| **BiLSTM** | `64 Units x 2` | 128 output dimensions (Bidirectional) |
| **Dense ReLU** | `64 Units` | Non-linear interaction + L2 Regularization |
| **Dropout** | `Rate = 0.30` | Prevents co-adaptation |
| **Output Sigmoid**| `1 Unit` | Cyberbullying Probability $[0, 1]$ |
        """)
        st.info("💡 **Why SpatialDropout1D?** Standard dropout drops individual random elements across the embedding matrix, which weakens words inconsistently. SpatialDropout1D drops entire feature channels across time-steps, forcing robust representation learning.")

# ----------------- TAB 3: Evaluation & Metrics -----------------
with tab3:
    st.subheader("📊 Performance Evaluation on Unseen Test Dataset")
    
    metrics_file = get_file_path("reports", "test_evaluation_report.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Overall Accuracy", f"{metrics_data.get('accuracy', 0)*100:.2f}%")
        m2.metric("Precision", f"{metrics_data.get('precision', 0)*100:.2f}%")
        m3.metric("Recall (Sensitivity)", f"{metrics_data.get('recall', 0)*100:.2f}%")
        m4.metric("F1-Score", f"{metrics_data.get('f1_score', 0)*100:.2f}%")
        m5.metric("ROC-AUC Score", f"{metrics_data.get('roc_auc', 0):.4f}")
    else:
        st.info("Evaluation metrics file found.")
        
    st.divider()
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        cm_img = get_file_path("reports", "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix on Test Dataset", use_container_width=True)
        else:
            st.info("Confusion matrix image pending.")
            
    with col_img2:
        roc_img = get_file_path("reports", "roc_curve.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Receiver Operating Characteristic (ROC-AUC) Curve", use_container_width=True)
        else:
            st.info("ROC Curve image pending.")
            
    tc_img = get_file_path("reports", "bilstm_training_curves.png")
    if os.path.exists(tc_img):
        st.subheader("📈 Training & Validation Convergence Curves")
        st.image(tc_img, caption="Loss and Accuracy convergence across epochs", use_container_width=True)

# ----------------- TAB 4: Dataset Explorer -----------------
with tab4:
    st.subheader("📁 Dataset Inspection & Class Distribution")
    data_csv = get_file_path("data", "cyberbullying_dataset.csv")
    if os.path.exists(data_csv):
        df_sample = pd.read_csv(data_csv)
        st.write(f"Total labeled samples in corpus: **{len(df_sample)}**")
        
        col_dist1, col_dist2 = st.columns([1, 2])
        with col_dist1:
            st.write("**Class Counts:**")
            counts = df_sample['label'].value_counts()
            st.dataframe(pd.DataFrame({
                "Class": ["Non-Cyberbullying (0)", "Cyberbullying (1)"],
                "Count": [counts.get(0, 0), counts.get(1, 0)]
            }))
        with col_dist2:
            st.write("**Random Data Samples:**")
            st.dataframe(df_sample.sample(min(8, len(df_sample))), use_container_width=True)
    else:
        st.info("Dataset file `cyberbullying_dataset.csv` not found.")
