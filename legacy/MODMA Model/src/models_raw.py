import tensorflow as tf
from tensorflow.keras import layers, models, constraints, regularizers

def EEGNet(nb_classes, Chans=128, Samples=128*4, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """
    Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024)
    
    Args:
        nb_classes: int, number of classes to classify.
        Chans: int, number of EEG channels.
        Samples: int, number of time points in each example.
        dropoutRate: float.
        kernLength: int, length of temporal convolution.
        F1: int, number of temporal filters.
        D: int, depth multiplier (number of spatial filters per temporal filter).
        F2: int, number of pointwise filters.
        norm_rate: float, max norm constraint.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
    
    input1 = layers.Input(shape=(Chans, Samples, 1))

    # Block 1
    # Temporal Convolution
    x = layers.Conv2D(F1, (1, kernLength), padding='same',
                      input_shape=(Chans, Samples, 1),
                      use_bias=False)(input1)
    x = layers.BatchNormalization()(x)
    
    # Depthwise Convolution (Spatial Filter)
    x = layers.DepthwiseConv2D((Chans, 1), use_bias=False, 
                               depth_multiplier=D,
                               depthwise_constraint=constraints.max_norm(1.))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = dropoutType(dropoutRate)(x)
    
    # Block 2
    # Separable Convolution
    x = layers.SeparableConv2D(F2, (1, 16),
                               use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = dropoutType(dropoutRate)(x)
    
    # Classification Block
    x = layers.Flatten()(x)
    
    if nb_classes == 1:
        # Binary Classification
        x = layers.Dense(1, activation='sigmoid', kernel_constraint=constraints.max_norm(norm_rate))(x)
    else:
        # Multi-class
        x = layers.Dense(nb_classes, activation='softmax', kernel_constraint=constraints.max_norm(norm_rate))(x)
    
    model = models.Model(inputs=input1, outputs=x)
    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

def TSception(nb_classes, Chans, Samples, sampling_rate, ch_input=128, num_T=9, num_S=6, hidden=128, dropout_rate=0.5):
    """
    TSception: A Multi-scale Differential Entropy Convolutional Neural Network for EEG Emotion Recognition
    Yi Ding et al., 2020.
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1))
    
    # Temporal Convolutions (Multi-scale)
    # Different kernel sizes to capture different frequency bands (Delta, Theta, Alpha, etc.)
    # Kernels: 0.5s, 0.25s, 0.125s (approx)
    
    # Using 3 scales
    scales = [int(sampling_rate * 0.5), int(sampling_rate * 0.25), int(sampling_rate * 0.125)]
    # Ensure scales are odd for symmetric padding if needed, but 'nice' convs handle it.
    
    t_layers = []
    for scale in scales:
        x = layers.Conv2D(num_T, (1, scale), padding='same', activation='relu')(input_layer)
        t_layers.append(x)
        
    x = layers.Concatenate(axis=-1)(t_layers)
    x = layers.BatchNormalization()(x)
    
    # Spatial Convolutions
    x = layers.Conv2D(num_S, (Chans, 1), padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Further processing
    x = layers.Flatten()(x)
    x = layers.Dense(hidden, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    if nb_classes == 1:
        output_layer = layers.Dense(1, activation='sigmoid')(x)
    else:
        output_layer = layers.Dense(nb_classes, activation='softmax')(x)
        
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

if __name__ == "__main__":
    # Test model build
    # EEGNet
    model = EEGNet(nb_classes=1, Chans=128, Samples=512)
    # TSception
    model2 = TSception(nb_classes=1, Chans=128, Samples=512, sampling_rate=128)
    model2.summary()
