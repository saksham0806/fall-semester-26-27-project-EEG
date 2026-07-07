import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Reshape, Permute, Concatenate,
    Conv2D, Embedding, LayerNormalization, Lambda, Flatten
)
from tensorflow.keras.models import Model

class CLSToken(tf.keras.layers.Layer):
    """Learnable [CLS] token prepended to patch sequence."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        cls_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcast, patches], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config

class AdaptiveInstanceNorm(tf.keras.layers.Layer):
    """
    Adaptive Instance Normalization conditioned on dataset_id.
    Normalizes each channel's spectrogram independently, then applies
    learned per-dataset affine transform.
    """
    def __init__(self, num_datasets=4, num_channels=6, style_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_datasets = num_datasets
        self.num_channels = num_channels
        self.style_dim = style_dim

    def build(self, input_shape):
        self.ds_embedding = Embedding(self.num_datasets, self.style_dim)
        self.gamma_dense = Dense(self.num_channels, use_bias=True)
        self.beta_dense = Dense(self.num_channels, use_bias=True)

    def call(self, inputs):
        spec, dataset_id = inputs
        # spec: (batch, F, T, C), dataset_id: (batch,)

        # Instance normalize per channel
        mean = tf.reduce_mean(spec, axis=[1, 2], keepdims=True)
        var = tf.math.reduce_variance(spec, axis=[1, 2], keepdims=True)
        spec_norm = (spec - mean) / tf.sqrt(var + 1e-5)

        # Generate style from dataset_id
        style = self.ds_embedding(dataset_id)              # (batch, style_dim)
        gamma = self.gamma_dense(style)                     # (batch, C)
        beta = self.beta_dense(style)                       # (batch, C)

        # Reshape for broadcasting: (batch, 1, 1, C)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.num_channels])
        beta = tf.reshape(beta, [-1, 1, 1, self.num_channels])

        return spec_norm * (1.0 + gamma) + beta

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_datasets': self.num_datasets,
            'num_channels': self.num_channels,
            'style_dim': self.style_dim,
        })
        return config

class SpectrogramLayer(tf.keras.layers.Layer):
    """Non-trainable layer that computes per-channel log-magnitude STFT."""
    def __init__(self, frame_length=64, frame_step=16, fft_length=128, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, x):
        # x: (batch, 6, 512)
        specs = []
        for ch in range(6):
            channel = x[:, ch, :]                                      # (batch, 512)
            stft = tf.signal.stft(
                channel,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length
            )                                                          # (batch, 29, 65)
            magnitude = tf.abs(stft)
            log_mag = tf.math.log(magnitude + 1e-6)
            specs.append(log_mag)

        stacked = tf.stack(specs, axis=1)                              # (batch, 6, 29, 65)
        return tf.transpose(stacked, perm=[0, 3, 2, 1])               # (batch, 65, 29, 6)

    def get_config(self):
        config = super().get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
        })
        return config

class TransformerEncoder(tf.keras.layers.Layer):
    """Single Transformer encoder layer with MHSA + FFN."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.mhsa = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn = self.mhsa(x, x)
        attn = self.drop1(attn, training=training)
        x = self.ln1(x + attn)

        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.ln2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

def build_st2adn(Chans=6, Samples=512, num_datasets=4,
                  embed_dim=128, num_heads=8, ff_dim=256,
                  num_layers=4, dropout_rate=0.3,
                  patch_freq=13, patch_time=5):
    """
    Spectro-Temporal Transformer with Adaptive Domain Normalization.

    Inputs:
        raw_eeg:    (batch, Chans, Samples, 1)
        dataset_id: (batch,) — integer 0-3

    Outputs:
        depression: (batch, 1)
        stress:     (batch, 1)
        freq_attention: (batch, 5) — interpretable frequency band weights
    """
    # --- Inputs ---
    raw_input = Input(shape=(Chans, Samples, 1), name='raw_eeg')
    ds_input = Input(shape=(), dtype=tf.int32, name='dataset_id')

    # --- Spectrogram ---
    x = Reshape((Chans, Samples), name='strip_dim')(raw_input)
    spec = SpectrogramLayer(name='stft')(x)             # (batch, 65, 29, 6)

    # --- Adaptive Instance Normalization ---
    spec_adain = AdaptiveInstanceNorm(
        num_datasets=num_datasets,
        num_channels=Chans,
        name='adain'
    )([spec, ds_input])                                  # (batch, 65, 29, 6)

    # --- Patch Embedding ---
    # (65, 29, 6) with patch (13, 5) → (5, 5, embed_dim) = 25 patches
    # Note: 29 / 5 = 5 remainder 4, use valid padding to drop edge
    patches = Conv2D(
        embed_dim, kernel_size=(patch_freq, patch_time),
        strides=(patch_freq, patch_time), padding='valid',
        activation='relu', name='patch_embed'
    )(spec_adain)                                        # (batch, 5, 5, 128)

    num_patches_f = 65 // patch_freq  # = 5
    num_patches_t = 29 // patch_time  # = 5
    num_patches = num_patches_f * num_patches_t  # = 25

    patches = Reshape((num_patches, embed_dim),
                       name='flatten_patches')(patches)  # (batch, 25, 128)

    # --- CLS Token ---
    tokens = CLSToken(embed_dim, name='cls_token')(patches)  # (batch, 26, 128)

    # --- Positional Embedding ---
    positions = tf.range(num_patches + 1)                # [0, 1, ..., 25]
    pos_embed = Embedding(num_patches + 1, embed_dim,
                          name='pos_embedding')(positions) # (26, 128)
    tokens = tokens + pos_embed                           # (batch, 26, 128)

    # --- Transformer Encoder Stack ---
    for i in range(num_layers):
        tokens = TransformerEncoder(
            embed_dim, num_heads, ff_dim, dropout_rate,
            name=f'transformer_{i}'
        )(tokens)

    # --- Extract CLS token ---
    cls_output = Lambda(
        lambda t: t[:, 0, :], name='extract_cls'
    )(tokens)                                             # (batch, 128)

    # --- Frequency Band Attention (interpretability) ---
    freq_attention = Dense(
        num_patches_f, activation='softmax',
        name='freq_band_attention'
    )(cls_output)                                         # (batch, 5)

    # --- Task Heads ---
    dep_h = Dense(64, activation='relu', name='dep_h1')(cls_output)
    dep_h = Dropout(0.4)(dep_h)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_h)

    str_h = Dense(64, activation='relu', name='str_h1')(cls_output)
    str_h = Dropout(0.4)(str_h)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_h)

    model = Model(
        inputs=[raw_input, ds_input],
        outputs=[dep_out, str_out, freq_attention],
        name='ST2_ADN'
    )
    return model
