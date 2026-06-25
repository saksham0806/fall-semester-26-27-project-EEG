import tensorflow as tf
from tensorflow.keras import layers, models, constraints

def _get_classification_layer(x, nb_classes, norm_rate=0.25):
    """Helper for BCI outputs with max-norm constraint support."""
    if nb_classes == 1:
        return layers.Dense(1, activation='sigmoid', kernel_constraint=constraints.max_norm(norm_rate), name='output')(x)
    else:
        return layers.Dense(nb_classes, activation='softmax', kernel_constraint=constraints.max_norm(norm_rate), name='output')(x)

def EEGNet(nb_classes, Chans=6, Samples=512, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    """
    Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024)
    adapted for 6-channel frontal EEG setups.
    
    Input shape: (Chans, Samples, 1)
    """
    input1 = layers.Input(shape=(Chans, Samples, 1), name='input')

    # Block 1
    # Temporal Convolution (learns frequency filters)
    x = layers.Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    x = layers.BatchNormalization()(x)
    
    # Depthwise Convolution (learns spatial filters for each temporal filter)
    x = layers.DepthwiseConv2D((Chans, 1), use_bias=False, 
                               depth_multiplier=D,
                               depthwise_constraint=constraints.max_norm(1.))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropoutRate)(x)
    
    # Block 2
    # Separable Convolution (summarizes individual feature maps)
    x = layers.SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropoutRate)(x)
    
    # Classification Block
    x = layers.Flatten()(x)
    output = _get_classification_layer(x, nb_classes, norm_rate)
    
    model = models.Model(inputs=input1, outputs=output)
    loss = 'binary_crossentropy' if nb_classes == 1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return model

def TSception(nb_classes, Chans=6, Samples=512, sampling_rate=128, num_T=9, num_S=6, hidden=128, dropout_rate=0.5):
    """
    TSception: A Multi-scale Differential Entropy Convolutional Neural Network for EEG
    adapted for 6-channel frontal setups.
    
    Input shape: (Chans, Samples, 1)
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')
    
    # Multi-scale Temporal Convolutions
    # Kernels capture 0.5s, 0.25s, 0.125s durations
    scales = [int(sampling_rate * 0.5), int(sampling_rate * 0.25), int(sampling_rate * 0.125)]
    
    t_layers = []
    for scale in scales:
        # Pad temporal convolutions to keep size consistent
        x = layers.Conv2D(num_T, (1, scale), padding='same', activation='relu')(input_layer)
        t_layers.append(x)
        
    x = layers.Concatenate(axis=-1)(t_layers)
    x = layers.BatchNormalization()(x)
    
    # Spatial Convolutions (integrate over all channels)
    x = layers.Conv2D(num_S, (Chans, 1), padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense classification head
    x = layers.Flatten()(x)
    x = layers.Dense(hidden, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    if nb_classes == 1:
        output_layer = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        output_layer = layers.Dense(nb_classes, activation='softmax', name='output')(x)
        
    model = models.Model(inputs=input_layer, outputs=output_layer)
    loss = 'binary_crossentropy' if nb_classes == 1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return model

class TransformerEncoderBlock(layers.Layer):
    """Standard Transformer Encoder block in Keras."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def EEGTransformer(nb_classes, Chans=6, Samples=512, patch_size=16, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2, dropout_rate=0.5):
    """
    EEG-Transformer model for raw EEG classification.
    Processes the raw signal by tokenizing it into non-overlapping temporal patches.
    
    Input shape: (Chans, Samples, 1)
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')
    
    # Reshape and Permute to (Samples, Chans) -> (512, 6)
    x = layers.Reshape((Chans, Samples))(input_layer)
    x = layers.Permute((2, 1))(x)
    
    # Tokenize the time axis using a 1D Conv layer as a patch projector
    # Resulting sequence length: Samples // patch_size (e.g. 512 // 16 = 32 tokens)
    x = layers.Conv1D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid', activation='relu')(x)
    
    seq_len = Samples // patch_size
    
    # Add Learned Positional Embedding
    pos_embeddings = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(tf.range(start=0, limit=seq_len, delta=1))
    x = x + pos_embeddings
    
    # Stack Transformer Encoder blocks
    for _ in range(num_layers):
        x = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)(x)
        
    # Pool across the sequence tokens
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    output_layer = _get_classification_layer(x, nb_classes)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    loss = 'binary_crossentropy' if nb_classes == 1 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return model
