"""
Evaluation and Metrics Generation Module for Cyberbullying Detection.
Computes Confusion Matrix, ROC-AUC Curve, and Classification Report on unseen test data.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf

def evaluate_cyberbullying_model(
    model_path: str,
    test_split_path: str,
    output_dir: str
):
    print(f"Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading test dataset from: {test_split_path}")
    test_data = np.load(test_split_path, allow_pickle=True)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    raw_texts = test_data['raw_texts']
    
    print(f"Evaluating {len(X_test)} unseen test samples...")
    y_pred_probs = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate Core Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Cyberbullying", "Cyberbullying"], output_dict=True)
    
    print("\n" + "="*50)
    print("           MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print("="*50)
    
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Plot Confusion Matrix
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Non-Cyberbullying (0)", "Cyberbullying (1)"],
        yticklabels=["Non-Cyberbullying (0)", "Cyberbullying (1)"],
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix on Test Dataset", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual Class (Ground Truth)", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to: {cm_path}")
    
    # 2. Plot ROC Curve
    roc_path = os.path.join(reports_dir, "roc_curve.png")
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'BiLSTM ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#d62728', lw=1.5, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC Curve saved to: {roc_path}")
    
    # 3. Save JSON Metrics
    metrics_summary = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    metrics_path = os.path.join(reports_dir, "test_evaluation_report.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Comprehensive metrics JSON saved to: {metrics_path}")
    
    # Display Sample Predictions
    print("\n--- Sample Test Predictions ---")
    for i in range(min(5, len(raw_texts))):
        status = "CORRECT" if y_test[i] == y_pred[i] else "INCORRECT"
        actual_label = "Cyberbullying" if y_test[i] == 1 else "Safe"
        pred_label = "Cyberbullying" if y_pred[i] == 1 else "Safe"
        print(f"[{status}] Prob: {y_pred_probs[i]:.4f} | Actual: {actual_label} | Pred: {pred_label}")
        print(f"  Text: \"{raw_texts[i]}\"\n")
        
    return metrics_summary

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(base_dir, "saved_models", "bilstm_model.keras")
    test_file = os.path.join(base_dir, "data", "test_split.npz")
    evaluate_cyberbullying_model(model_file, test_file, base_dir)
