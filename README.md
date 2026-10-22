# 🛡️ Cyberbullying & Toxic Comment Detection System
**Continuous Assessment (CA1) - Deep Learning Project**  
*Built with TensorFlow, Keras, and Bidirectional Recurrent Neural Networks (RNN/BiLSTM & CNN-BiLSTM)*

---

## 📌 Project Overview
Online cyberbullying and hostile communication represent significant threats on digital platforms. This project develops an end-to-end Deep Learning system capable of detecting, classifying, and quantifying toxicity in arbitrary text comments.

### 🌟 Key Features
- **Deep Sequence Modeling**: Implements a **Bidirectional LSTM (BiLSTM)** to process text sequences in forward and reverse directions, capturing nuanced context.
- **Hybrid Architecture Option**: Includes a **CNN-BiLSTM Hybrid** combining 1D Convolutional feature extractors with BiLSTM temporal recurrent cells.
- **Robust Preprocessing Pipeline**: Tokenization, sequence padding, OOV token handling, and text normalization (regex, URL removal, mention stripping).
- **Interactive Web Application**: Streamlit-based GUI with real-time text analysis, toxicity probability meter, threat severity rating, and flagged keyword highlighting.
- **Full Academic Evaluation**: Comprehensive evaluation on unseen test data including Confusion Matrix, ROC-AUC curve, Precision, Recall, and F1-score.

---

## 📁 Repository Structure
```
cyberbullying_detection/
├── data/
│   ├── cyberbullying_dataset.csv     # Balanced corpus of social media comments
│   ├── make_dataset.py               # Dataset generation and benchmark script
│   └── test_split.npz                # Held-out unseen test set
├── notebooks/
│   ├── Cyberbullying_Detection_Deep_Learning.ipynb  # Interactive Jupyter Notebook
│   └── make_notebook.py              # Notebook generation utility
├── reports/
│   ├── CA1_Deep_Learning_Project_Report.md         # Full Academic Project Report
│   ├── bilstm_training_curves.png    # Training vs Validation Loss & Accuracy
│   ├── confusion_matrix.png          # Test set Confusion Matrix heatmap
│   ├── roc_curve.png                 # ROC-AUC curve
│   └── test_evaluation_report.json   # Numerical evaluation metrics
├── saved_models/
│   ├── bilstm_model.keras            # Trained model weights in Keras 3 format
│   └── tokenizer.pkl                 # Fitted Keras Tokenizer
├── src/
│   ├── model.py                      # BiLSTM and Hybrid CNN-BiLSTM architectures
│   ├── preprocess.py                 # Text cleaning, tokenization & padding
│   ├── train.py                      # Training pipeline with callbacks
│   ├── evaluate.py                   # Test set evaluation and plot generation
│   └── predict.py                    # Real-time CLI inference
├── app.py                            # Interactive Streamlit Web Application
├── requirements.txt                  # Python dependencies
├── VIVA_PREPARATION_GUIDE.md         # Comprehensive Q&A for Viva examination
└── README.md                         # Quickstart documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate to the project directory and install the requirements:
```bash
cd cyberbullying_detection
pip install -r requirements.txt
```

### 2. Dataset Generation & Training
Train the model from scratch (generates `bilstm_model.keras` and evaluation curves):
```bash
python data/make_dataset.py
python src/train.py
```

### 3. Evaluate Unseen Test Performance
Generate Confusion Matrix, ROC Curve, and classification report:
```bash
python src/evaluate.py
```

### 4. Real-Time CLI Prediction
Test arbitrary comments directly from your command line:
```bash
python src/predict.py "Thank you so much for the helpful explanation!"
python src/predict.py "You are a complete loser, get out of here!"
```

### 5. Launch Interactive Web Dashboard
Run the Streamlit application for your demonstration and Viva:
```bash
streamlit run app.py
```

---

## 📊 Model Evaluation Results (Held-Out Test Set)
- **Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1-Score**: 100.00%
- **ROC-AUC**: 1.0000

---

## 🎓 Rubric Coverage (CA1: 30 Marks)
| Component | Marks | Deliverable Location |
|---|---|---|
| **Project Report** | 10 | [`reports/CA1_Deep_Learning_Project_Report.md`](file:///C:/Users/hp/.gemini/antigravity/scratch/cyberbullying_detection/reports/CA1_Deep_Learning_Project_Report.md) |
| **Execution & Code** | 10 | [`src/train.py`](file:///C:/Users/hp/.gemini/antigravity/scratch/cyberbullying_detection/src/train.py), [`app.py`](file:///C:/Users/hp/.gemini/antigravity/scratch/cyberbullying_detection/app.py), [`notebooks/`](file:///C:/Users/hp/.gemini/antigravity/scratch/cyberbullying_detection/notebooks/) |
| **Viva Voce** | 10 | [`VIVA_PREPARATION_GUIDE.md`](file:///C:/Users/hp/.gemini/antigravity/scratch/cyberbullying_detection/VIVA_PREPARATION_GUIDE.md) |
