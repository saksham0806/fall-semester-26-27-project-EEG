import tensorflow as tf
from tensorflow.keras import layers, models

def _get_classification_layer(x, nb_classes):
    """Helper to build output layer based on class count."""
    if nb_classes == 1:
        return layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        return layers.Dense(nb_classes, activation='softmax', name='output')(x)

def build_1d_cnn(nb_classes, Chans=6, Samples=512, dropout_rate=0.5):
    """
    Build a 1D Convolutional Neural Network for raw EEG time-series.
    Input shape is (Chans, Samples, 1). Internally transposed to (Samples, Chans)
    to run 1D convolutions over time.
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')
    
    # Preprocess: reshape to (Chans, Samples) and permute to (Samples, Chans)
    x = layers.Reshape((Chans, Samples))(input_layer)
    x = layers.Permute((2, 1))(x) # Shape: (Samples, Chans)
    
    # Conv Block 1
    x = layers.Conv1D(filters=32, kernel_size=10, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Conv Block 2
    x = layers.Conv1D(filters=64, kernel_size=10, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Conv Block 3
    x = layers.Conv1D(filters=128, kernel_size=10, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    output_layer = _get_classification_layer(x, nb_classes)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    loss = 'binary_crossentropy' if nb_classes == 1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def build_lstm(nb_classes, Chans=6, Samples=512, dropout_rate=0.5):
    """
    Build an LSTM model for raw EEG time-series.
    Input shape is (Chans, Samples, 1). Internally transposed to (Samples, Chans).
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')
    
    # Reshape and Permute to (Samples, Chans)
    x = layers.Reshape((Chans, Samples))(input_layer)
    x = layers.Permute((2, 1))(x)
    
    # Recurrent Layers
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.LSTM(64)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    output_layer = _get_classification_layer(x, nb_classes)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    loss = 'binary_crossentropy' if nb_classes == 1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def build_cnn_lstm(nb_classes, Chans=6, Samples=512, dropout_rate=0.5):
    """
    Build a hybrid CNN-LSTM model for raw EEG time-series.
    Input shape is (Chans, Samples, 1). Internally transposed to (Samples, Chans).
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')
    
    # Reshape and Permute to (Samples, Chans)
    x = layers.Reshape((Chans, Samples))(input_layer)
    x = layers.Permute((2, 1))(x)
    
    # CNN spatial-temporal features extraction
    x = layers.Conv1D(filters=32, kernel_size=10, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=10, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers for temporal tracking
    x = layers.LSTM(64)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    output_layer = _get_classification_layer(x, nb_classes)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    loss = 'binary_crossentropy' if nb_classes == 1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model
