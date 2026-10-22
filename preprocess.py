"""
Text Preprocessing and Sequence Formatting Pipeline for Cyberbullying Detection.
"""

import re
import os
import string
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DEFAULT_VOCAB_SIZE = 10000
DEFAULT_MAX_LEN = 100
OOV_TOKEN = "<OOV>"

def clean_text(text: str) -> str:
    """
    Cleans raw social media text:
    - Lowercases text
    - Removes URLs and hyperlinks
    - Removes user handles / mentions (@username)
    - Removes HTML tags
    - Cleans punctuation and numbers
    - Normalizes multiple spaces
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fit_tokenizer(texts: List[str], max_vocab: int = DEFAULT_VOCAB_SIZE, oov_token: str = OOV_TOKEN) -> Tokenizer:
    """
    Fits Keras Tokenizer on cleaned text corpus.
    """
    tokenizer = Tokenizer(num_words=max_vocab, oov_token=oov_token, lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_padded_sequences(texts: List[str], tokenizer: Tokenizer, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """
    Converts list of text strings into padded numeric sequences.
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

def save_tokenizer(tokenizer: Tokenizer, save_path: str):
    """
    Saves the fitted tokenizer to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to: {save_path}")

def load_tokenizer(load_path: str) -> Tokenizer:
    """
    Loads fitted tokenizer from disk.
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    sample_text = "Hey @user check this out https://example.com!! You are so stupid and useless... #hate"
    cleaned = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned:  {cleaned}")
