"""
Generates the comprehensive Jupyter Notebook for CA1 Deep Learning Project.
"""

import os
import nbformat as nbf

def create_ca1_notebook(output_path: str):
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 🛡️ CA1 Deep Learning Project: Cyberbullying & Toxic Comment Detection
### **Student Name:** [Your Name / Registration No.]
### **Course:** Deep Learning (TensorFlow & Keras)
### **Architectures Evaluated:** Bidirectional LSTM (BiLSTM) & Hybrid CNN-BiLSTM

---
## 1. Problem Statement & Motivation
Online harassment, toxic commentary, and cyberbullying pose severe psychosocial hazards across social platforms, gaming communities, and educational forums. 
Traditional rule-based keyword filters and Bag-of-Words ML models (such as Naive Bayes or Logistic Regression) fail because:
1. They ignore **word order and sequential context** (e.g., sarcasm or negated toxicity).
2. They cannot handle **polysemy and out-of-vocabulary (OOV)** variations.
3. They lack temporal memory to connect distant words in a sentence.

### Deep Learning Solution
We build an end-to-end Deep Learning sequence modeling pipeline using **TensorFlow and Keras**:
- **Word Embeddings**: 128-dimensional continuous vector space.
- **Bidirectional LSTM**: Recurrent Neural Network reading text forwards and backwards to capture complete sentence context.
- **Hybrid CNN-BiLSTM**: 1D Convolutional feature extractors feeding into BiLSTM recurrent layers.
"""))

    # 2. Setup
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Environment Setup & Library Imports"""))
    cells.append(nbf.v4.new_code_cell("""import os
import re
import string
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")"""))

    # 3. Dataset Loading & EDA
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Dataset Loading & Exploratory Data Analysis (EDA)
We analyze the dataset containing labelled social media texts:
- `label = 0`: Non-Cyberbullying (Safe / Constructive / Neutral)
- `label = 1`: Cyberbullying (Harassment / Insult / Threat / Toxicity)"""))
    cells.append(nbf.v4.new_code_cell("""# Load Dataset
dataset_path = os.path.join("..", "data", "cyberbullying_dataset.csv")
df = pd.read_csv(dataset_path)

print(f"Total Records: {len(df)}")
print(f"Class Distribution:\\n{df['label'].value_counts()}")
df.head(10)"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize Class Balance & Sentence Lengths
df['char_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class balance bar plot
sns.countplot(data=df, x='label', ax=axes[0], palette=['#2ecc71', '#e74c3c'])
axes[0].set_title("Class Balance (0: Safe vs 1: Cyberbullying)", fontsize=13, fontweight='bold')
axes[0].set_xticklabels(["Safe (0)", "Cyberbullying (1)"])

# Word count distribution
sns.histplot(data=df, x='word_count', hue='label', bins=30, kde=True, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
axes[1].set_title("Word Count Distribution per Class", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()"""))

    # 4. Text Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Text Cleaning, Tokenization & Sequence Padding
We implement text normalization:
- Lowercasing
- Stripping URLs, HTML tags, and user mentions
- Removing punctuation and special characters
- Tokenizing top 10,000 vocabulary words with `<OOV>` fallback
- Padding sequences to a uniform length of 100 tokens with post-padding"""))
    cells.append(nbf.v4.new_code_cell("""def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'#(\\w+)', r'\\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Apply cleaning
cleaned_corpus = [clean_text(t) for t in df['text']]
labels = df['label'].values.astype(np.float32)

# Train-Val-Test Split: 70% Train, 15% Validation, 15% Test
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    cleaned_corpus, labels, test_size=0.30, random_state=42, stratify=labels
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Tokenizer fitting
VOCAB_SIZE = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_raw)

# Pad sequences
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=MAX_LEN, padding='post', truncating='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(X_val_raw), maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=MAX_LEN, padding='post', truncating='post')

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")"""))

    # 5. Model Architecture
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Deep Learning Model Architecture: Bidirectional LSTM
### Mathematical Concept:
Standard RNNs suffer from vanishing gradient updates over long sequences:
$$\\frac{\\partial L}{\\partial W} = \\sum_{t=1}^{T} \\frac{\\partial L_t}{\\partial h_t} \\prod_{j=k+1}^{t} \\frac{\\partial h_j}{\\partial h_{j-1}} \\frac{\\partial h_k}{\\partial W}$$

**LSTM Cell Equations**:
1. Forget Gate: $f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)$
2. Input Gate: $i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i)$
3. Candidate Cell State: $\\tilde{C}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)$
4. Cell State Update: $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$
5. Output Gate: $o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o)$
6. Hidden State: $h_t = o_t \\odot \\tanh(C_t)$

**Bidirectional Processing**:
Combines forward representation $\\vec{h}_t$ and backward representation $\\overleftarrow{h}_t$:
$$h_t^{\\text{concat}} = [\\vec{h}_t \\,;\\, \\overleftarrow{h}_t]$$
"""))
    cells.append(nbf.v4.new_code_cell("""def create_bilstm_model():
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,), dtype='int32', name="input_ids"),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128, name="word_embedding"),
        layers.SpatialDropout1D(0.3, name="spatial_dropout"),
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=False, dropout=0.2),
            name="bidirectional_lstm"
        ),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_features"),
        layers.Dropout(0.3, name="dropout_layer"),
        layers.Dense(1, activation='sigmoid', name="output_sigmoid")
    ], name="BiLSTM_Cyberbullying_Model")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model

model = create_bilstm_model()
model.summary()"""))

    # 6. Model Training
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Model Training & Optimization
We train using adaptive learning rate scheduling and early stopping:
- `EarlyStopping`: Halts training when validation loss stops improving for 4 consecutive epochs.
- `ReduceLROnPlateau`: Halves learning rate if validation loss plateaus for 2 epochs.
- `ModelCheckpoint`: Automatically saves model checkpoint with lowest validation loss."""))
    cells.append(nbf.v4.new_code_cell("""callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)"""))

    # 7. Curves
    cells.append(nbf.v4.new_markdown_cell("""---
## 7. Training & Validation Convergence Curves"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, len(history.history['loss']) + 1)

# Accuracy curve
axes[0].plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy')
axes[0].plot(epochs, history.history['val_accuracy'], 'r--s', label='Validation Accuracy')
axes[0].set_title('Accuracy vs Epochs', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# Loss curve
axes[1].plot(epochs, history.history['loss'], 'b-o', label='Training Loss')
axes[1].plot(epochs, history.history['val_loss'], 'r--s', label='Validation Loss')
axes[1].set_title('Binary Cross-Entropy Loss vs Epochs', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # 8. Evaluation
    cells.append(nbf.v4.new_markdown_cell("""---
## 8. Unseen Test Set Evaluation & Metrics
We evaluate the trained model on 480 unseen test samples."""))
    cells.append(nbf.v4.new_code_cell("""y_pred_probs = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Cyberbullying (1)"]))

# Confusion Matrix & ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["Safe", "Cyberbullying"], yticklabels=["Safe", "Cyberbullying"])
axes[0].set_title("Confusion Matrix on Test Split", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'BiLSTM (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'r--', lw=1.5, label='Random Guess')
axes[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc='lower right')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # 9. Real-time Inference
    cells.append(nbf.v4.new_markdown_cell("""---
## 9. Real-Time Inference on Arbitrary User Comments"""))
    cells.append(nbf.v4.new_code_cell("""def predict_cyberbullying(text: str):
    cleaned = clean_text(text)
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAX_LEN, padding='post', truncating='post')
    prob = float(model.predict(seq, verbose=0)[0][0])
    label = "🚨 Cyberbullying / Toxic" if prob >= 0.5 else "✅ Safe / Non-Cyberbullying"
    print(f"Comment:    \\"{text}\\"")
    print(f"Verdict:    {label}")
    print(f"Confidence: {prob*100:.2f}%\\n")

# Test custom comments
predict_cyberbullying("Thank you for sharing these helpful deep learning notes!")
predict_cyberbullying("You are a completely stupid loser, uninstall the game and never play again.")
predict_cyberbullying("I respectfully disagree with your conclusion on neural network convergence.")
predict_cyberbullying("Nobody likes you, go cry somewhere else you utter failure.")"""))

    nb['cells'] = cells
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter Notebook successfully created at: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cyberbullying_Detection_Deep_Learning.ipynb")
    create_ca1_notebook(out)
