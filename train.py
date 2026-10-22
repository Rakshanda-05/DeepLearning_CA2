"""
Training Pipeline for Deep Learning Cyberbullying Detection Model.
Trains BiLSTM and Hybrid CNN-BiLSTM models, tracks metrics, and generates loss/accuracy plots.
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocess import (
    clean_text,
    fit_tokenizer,
    texts_to_padded_sequences,
    save_tokenizer,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_MAX_LEN
)
from model import build_bilstm_model, build_cnn_bilstm_hybrid_model, compile_model

def plot_training_history(history, save_path: str):
    """
    Plots training and validation accuracy and loss curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r--s', label='Validation Accuracy')
    plt.title('Model Accuracy vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r--s', label='Validation Loss')
    plt.title('Model Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")

def train_cyberbullying_model(
    data_path: str,
    output_dir: str,
    architecture: str = "bilstm",
    epochs: int = 10,
    batch_size: int = 32,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_len: int = DEFAULT_MAX_LEN
):
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")
    
    # Preprocessing
    print("Preprocessing texts...")
    cleaned_texts = [clean_text(t) for t in df['text']]
    labels = df['label'].values.astype(np.float32)
    
    # Train / Val / Test split: 70% / 15% / 15%
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        cleaned_texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Splits -> Train: {len(X_train_raw)}, Validation: {len(X_val_raw)}, Test: {len(X_test_raw)}")
    
    # Fit Tokenizer only on Train data (prevents data leakage)
    print("Fitting tokenizer on training corpus...")
    tokenizer = fit_tokenizer(X_train_raw, max_vocab=vocab_size)
    
    # Convert to padded sequences
    X_train = texts_to_padded_sequences(X_train_raw, tokenizer, max_len=max_len)
    X_val = texts_to_padded_sequences(X_val_raw, tokenizer, max_len=max_len)
    X_test = texts_to_padded_sequences(X_test_raw, tokenizer, max_len=max_len)
    
    # Save tokenizer and test set for standalone evaluation
    models_dir = os.path.join(output_dir, "saved_models")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(models_dir, "tokenizer.pkl")
    save_tokenizer(tokenizer, tokenizer_path)
    
    # Save test set for evaluate.py
    test_data_path = os.path.join(output_dir, "data", "test_split.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test, raw_texts=X_test_raw)
    print(f"Test split saved to: {test_data_path}")
    
    # Build chosen model
    print(f"Building {architecture.upper()} architecture...")
    if architecture.lower() == "hybrid":
        model = build_cnn_bilstm_hybrid_model(vocab_size=vocab_size, max_length=max_len)
        model_name = "hybrid_cnn_bilstm_model.keras"
    else:
        model = build_bilstm_model(vocab_size=vocab_size, max_length=max_len)
        model_name = "bilstm_model.keras"
        
    compile_model(model)
    model.summary()
    
    best_model_path = os.path.join(models_dir, model_name)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\nStarting Deep Learning Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot and save curves
    plot_training_history(history, os.path.join(reports_dir, f"{architecture}_training_curves.png"))
    
    # Final evaluation on unseen test split
    print("\nEvaluating best model on unseen test dataset...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    metric_names = model.metrics_names
    eval_dict = dict(zip(metric_names, [float(v) for v in test_results]))
    
    print("\n--- Test Set Performance ---")
    for k, v in eval_dict.items():
        print(f"  {k.upper()}: {v:.4f}")
        
    eval_summary_path = os.path.join(reports_dir, f"{architecture}_test_metrics.json")
    with open(eval_summary_path, 'w') as f:
        json.dump(eval_dict, f, indent=4)
        
    print(f"\nTraining pipeline completed successfully! Model saved to: {best_model_path}")
    return model, history, eval_dict

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, "data", "cyberbullying_dataset.csv")
    train_cyberbullying_model(data_path=data_csv, output_dir=base_dir, architecture="bilstm", epochs=8)
