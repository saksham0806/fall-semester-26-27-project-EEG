import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape):
    """
    Build a 1D CNN model for EEG classification.
    
    Args:
        input_shape (tuple): Shape of the input data (n_time_steps, n_channels).
        
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model = models.Sequential()
    
    # First Conv Block
    model.add(layers.Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Second Conv Block
    model.add(layers.Conv1D(filters=64, kernel_size=10, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Third Conv Block (Optional, deeper for complex features)
    model.add(layers.Conv1D(filters=128, kernel_size=10, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    
    # Dense Layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output Layer (Binary Classification)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile Model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    
    return model

if __name__ == "__main__":
    # Test model build
    dummy_shape = (1000, 3)
    model = build_cnn_model(dummy_shape)
    model.summary()
