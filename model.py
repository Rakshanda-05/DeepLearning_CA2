"""
Deep Learning Model Architectures for Cyberbullying Detection.
Includes:
1. Bidirectional LSTM (BiLSTM) Architecture (RNN-based)
2. Hybrid CNN-BiLSTM Architecture (Combines 1D CNN feature extraction + Bidirectional LSTM sequence modeling)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_bilstm_model(
    vocab_size: int = 10000,
    embedding_dim: int = 128,
    max_length: int = 100,
    lstm_units: int = 64,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Constructs a Bidirectional LSTM Deep Learning Architecture.
    
    Layers:
    1. Input Layer: Fixed sequence length of token IDs
    2. Embedding Layer: Learns continuous dense word vectors
    3. SpatialDropout1D: Drops 1D feature maps to prevent feature co-adaptation
    4. Bidirectional LSTM: Captures forward and backward sequence dynamics
    5. Dense Layer (ReLU): Learns non-linear combination of hidden representations
    6. Dropout: Regularization to minimize overfitting
    7. Dense Layer (Sigmoid): Outputs cyberbullying probability [0, 1]
    """
    model = models.Sequential([
        layers.Input(shape=(max_length,), dtype='int32', name="input_sequence"),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="word_embedding"
        ),
        layers.SpatialDropout1D(dropout_rate, name="spatial_dropout"),
        layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.0),
            name="bidirectional_lstm"
        ),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_features"),
        layers.Dropout(dropout_rate, name="dense_dropout"),
        layers.Dense(1, activation='sigmoid', name="output_probability")
    ], name="BiLSTM_Cyberbullying_Classifier")
    
    return model

def build_cnn_bilstm_hybrid_model(
    vocab_size: int = 10000,
    embedding_dim: int = 128,
    max_length: int = 100,
    num_filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 64,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Constructs a Hybrid CNN-BiLSTM Deep Learning Architecture.
    
    Combines:
    1. Input Layer: Fixed sequence length of token IDs
    2. Embedding Layer: Learns continuous dense word vectors
    3. SpatialDropout1D: Prevents embedding co-adaptation
    4. Conv1D: Extracts local n-gram semantic patterns across adjacent words
    5. MaxPooling1D: Downsamples spatial dimensions while preserving salient features
    6. Bidirectional LSTM: Models sequence dependencies across convolutional features
    7. Dense + Dropout: High-level classification features
    8. Sigmoid Output: Probability of cyberbullying
    """
    model = models.Sequential([
        layers.Input(shape=(max_length,), dtype='int32', name="input_sequence"),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="word_embedding"
        ),
        layers.SpatialDropout1D(dropout_rate, name="spatial_dropout"),
        layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name="conv1d_feature_extractor"
        ),
        layers.MaxPooling1D(pool_size=2, name="max_pooling1d"),
        layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.0),
            name="bidirectional_lstm"
        ),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_layer"),
        layers.Dropout(dropout_rate, name="dense_dropout"),
        layers.Dense(1, activation='sigmoid', name="output_probability")
    ], name="Hybrid_CNN_BiLSTM_Classifier")
    
    return model

def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Compiles the model with Adam optimizer, binary cross-entropy, and comprehensive metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

if __name__ == "__main__":
    bilstm = build_bilstm_model()
    compile_model(bilstm)
    bilstm.summary()
    
    hybrid = build_cnn_bilstm_hybrid_model()
    compile_model(hybrid)
    hybrid.summary()
