"""
Inference and Real-Time Prediction Module for Cyberbullying Detection.
Takes arbitrary social media comments and outputs toxicity probability and classification.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from preprocess import clean_text, texts_to_padded_sequences, load_tokenizer, DEFAULT_MAX_LEN

# Lexicon of known hostile/flagged terms for hazard highlight
TOXIC_KEYWORDS = {
    "stupid", "idiot", "loser", "ugly", "freak", "disgusting", "pathetic",
    "trash", "worthless", "hate", "clown", "failure", "kill", "die", "shut up",
    "dumbest", "useless", "scumbag", "brainless", "jump", "suffering", "garbage"
}

class CyberbullyingPredictor:
    def __init__(self, model_path: str, tokenizer_path: str, max_len: int = DEFAULT_MAX_LEN):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}.")
            
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_len = max_len

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Analyzes a single input text and produces structured prediction results.
        """
        cleaned = clean_text(raw_text)
        padded_seq = texts_to_padded_sequences([raw_text], self.tokenizer, max_len=self.max_len)
        
        prob = float(self.model.predict(padded_seq, verbose=0)[0][0])
        is_cyberbullying = bool(prob >= 0.5)
        
        # Determine Severity Level
        if prob < 0.25:
            severity = "Low Risk (Safe)"
            badge_color = "green"
        elif prob < 0.50:
            severity = "Moderate (Neutral / Mild)"
            badge_color = "blue"
        elif prob < 0.80:
            severity = "High Risk (Toxic)"
            badge_color = "orange"
        else:
            severity = "Critical Risk (Severe Harassment / Threat)"
            badge_color = "red"
            
        # Detect flagged keywords
        words_in_text = cleaned.split()
        flagged_words = [w for w in words_in_text if w in TOXIC_KEYWORDS]
        
        return {
            "input_text": raw_text,
            "cleaned_text": cleaned,
            "probability": prob,
            "confidence_pct": round(prob * 100, 2),
            "label": "Cyberbullying / Toxic" if is_cyberbullying else "Non-Cyberbullying / Safe",
            "is_toxic": is_cyberbullying,
            "severity": severity,
            "badge_color": badge_color,
            "flagged_keywords": flagged_words
        }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "saved_models", "bilstm_model.keras")
    tokenizer_path = os.path.join(base_dir, "saved_models", "tokenizer.pkl")
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "You are an absolute loser, nobody wants you here in this group!"
        
    predictor = CyberbullyingPredictor(model_path, tokenizer_path)
    res = predictor.predict(query)
    
    print("\n--- Cyberbullying Detection Result ---")
    print(f"Input:       {res['input_text']}")
    print(f"Verdict:     {res['label']}")
    print(f"Confidence:  {res['confidence_pct']}%")
    print(f"Severity:    {res['severity']}")
    print(f"Flagged:     {res['flagged_keywords']}")
