import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, BatchNormalization, Dropout, Conv1D, GlobalAveragePooling1D, Flatten, MultiHeadAttention, LayerNormalization, Add

def build_lstm(input_shape, num_classes):
    """Enhanced Bidirectional LSTM with Attention"""
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Self-Attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attention_output = Dropout(0.2)(attention_output)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Third LSTM layer
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_cnn(input_shape, num_classes):
    """Enhanced 1D CNN with Residual Connections"""
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Block 2 with residual
    residual = Conv1D(filters=128, kernel_size=1, padding='same')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    
    # Block 3
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_mlp(input_shape, num_classes):
    """Enhanced Multi-Layer Perceptron"""
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def transformer_encoder_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    """Single Transformer Encoder Block"""
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    x1 = Add()([x, attention_output])
    x1 = LayerNormalization(epsilon=1e-6)(x1)
    
    # Feed Forward Network
    x2 = Dense(ff_dim, activation="relu")(x1)
    x2 = Dense(x.shape[-1])(x2)
    x2 = Dropout(dropout_rate)(x2)
    x = Add()([x1, x2])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x

def build_transformer(input_shape, num_classes):
    """Enhanced Transformer with Multiple Encoder Blocks"""
    inputs = Input(shape=input_shape)
    
    # Project input to higher dimension
    x = Dense(64)(inputs)
    
    # Transformer Encoder Block 1
    x = transformer_encoder_block(x, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1)
    
    # Transformer Encoder Block 2
    x = transformer_encoder_block(x, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1)
    
    # Transformer Encoder Block 3
    x = transformer_encoder_block(x, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1)
    
    # Global Pooling and Output
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
