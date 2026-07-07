import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Flatten,
    Conv1D, MaxPooling1D, LSTM, Bidirectional,
    Reshape, Permute, Multiply, RepeatVector,
    Lambda, LayerNormalization
)
from tensorflow.keras.models import Model

def _compute_channel_stats(x):
    """
    x: (batch, 512, 6)
    Returns: (batch, 30) — 5 stats per channel
    """
    mean = tf.reduce_mean(x, axis=1)                      # (batch, 6)
    var = tf.math.reduce_variance(x, axis=1)               # (batch, 6)
    max_val = tf.reduce_max(x, axis=1)                     # (batch, 6)
    min_val = tf.reduce_min(x, axis=1)                     # (batch, 6)
    energy = tf.reduce_sum(x ** 2, axis=1)                 # (batch, 6)
    return tf.concat([mean, var, max_val, min_val, energy], axis=-1)  # (batch, 30)

def _build_scale_branch(x, conv_kernels, pool_sizes, lstm_units, name_prefix):
    """
    Builds one temporal scale branch: Conv1D blocks → BiLSTM.

    Args:
        x: (batch, 512, 6) — channel-masked input
        conv_kernels: list of 2 kernel sizes, e.g. [16, 32]
        pool_sizes: list of 2 pool sizes, e.g. [2, 2]
        lstm_units: int, e.g. 32
        name_prefix: string for unique layer naming
    Returns:
        (batch, lstm_units*2) — BiLSTM output
    """
    h = Conv1D(32, conv_kernels[0], activation='relu', padding='same',
               name=f'{name_prefix}_conv1')(x)
    h = BatchNormalization(name=f'{name_prefix}_bn1')(h)
    h = MaxPooling1D(pool_sizes[0], name=f'{name_prefix}_pool1')(h)

    h = Conv1D(32, conv_kernels[1], activation='relu', padding='same',
               name=f'{name_prefix}_conv2')(h)
    h = BatchNormalization(name=f'{name_prefix}_bn2')(h)
    h = MaxPooling1D(pool_sizes[1], name=f'{name_prefix}_pool2')(h)

    h = Bidirectional(
        LSTM(lstm_units), name=f'{name_prefix}_bilstm'
    )(h)
    return h

def build_hamsnet(Chans=6, Samples=512, lstm_units=32,
                  shared_dim=128, dropout_rate=0.4):
    """
    Hierarchical Attention-Gated Multi-Scale Network (HAMS-Net).

    Input:  raw_eeg — (batch, Chans, Samples, 1)
    Outputs: depression (batch, 1), stress (batch, 1)
    """
    raw_input = Input(shape=(Chans, Samples, 1), name='raw_eeg')

    # --- Reshape: (batch, 6, 512, 1) → (batch, 512, 6) ---
    x = Reshape((Chans, Samples), name='strip_channel_dim')(raw_input)
    x = Permute((2, 1), name='to_time_first')(x)  # (batch, 512, 6)

    # --- Channel Reliability Gate ---
    ch_stats = Lambda(_compute_channel_stats, name='channel_stats')(x)  # (batch, 30)
    reliability = Dense(Chans, activation='sigmoid',
                        name='channel_reliability')(ch_stats)            # (batch, 6)
    r_expanded = RepeatVector(Samples, name='expand_reliability')(reliability)  # (batch, 512, 6)
    x_masked = Multiply(name='apply_channel_mask')([x, r_expanded])     # (batch, 512, 6)

    # --- Multi-Scale Branches ---
    s1 = _build_scale_branch(x_masked, conv_kernels=[16, 32],
                             pool_sizes=[2, 2], lstm_units=lstm_units,
                             name_prefix='fine')      # (batch, 64)

    s2 = _build_scale_branch(x_masked, conv_kernels=[64, 128],
                             pool_sizes=[2, 2], lstm_units=lstm_units,
                             name_prefix='mid')       # (batch, 64)

    s3 = _build_scale_branch(x_masked, conv_kernels=[128, 256],
                             pool_sizes=[4, 2], lstm_units=lstm_units,
                             name_prefix='coarse')    # (batch, 64)

    # --- Cross-Scale Attention ---
    # Stack: (batch, 3, 64)
    S = Lambda(lambda args: tf.stack(args, axis=1),
               name='stack_scales')([s1, s2, s3])

    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=16, name='cross_scale_attn'
    )(S, S)
    S = LayerNormalization(name='scale_norm')(S + attn_out)
    scale_features = Flatten(name='scale_flatten')(S)  # (batch, 192)

    # --- Shared Representation ---
    shared = Dense(shared_dim, activation='relu', name='shared_dense')(scale_features)
    shared = BatchNormalization(name='shared_bn')(shared)
    shared = Dropout(dropout_rate, name='shared_dropout')(shared)

    # --- Task Heads ---
    dep_h = Dense(32, activation='relu', name='dep_hidden')(shared)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_h)

    str_h = Dense(32, activation='relu', name='str_hidden')(shared)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_h)

    return Model(inputs=raw_input, outputs=[dep_out, str_out], name='HAMS_Net')
