import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout,
    Concatenate, Multiply, Lambda
)
from tensorflow.keras.models import Model
from ..proposed_model.eeg_models import TSception

def build_dsffn(Chans=6, Samples=512, sampling_rate=128,
                deep_proj_dim=128, hc_proj_dim=64, shared_dim=128,
                dropout_rate=0.4):
    """
    Dual-Stream Feature-Fusion Network.

    Inputs:
        raw_eeg:              (batch, Chans, Samples, 1)
        handcrafted_features: (batch, 81)

    Outputs:
        depression: (batch, 1) — sigmoid
        stress:     (batch, 1) — sigmoid
    """
    # --- Input layers ---
    raw_input = Input(shape=(Chans, Samples, 1), name='raw_eeg')
    hc_input = Input(shape=(81,), name='handcrafted_features')

    # --- Stream 1: TSception backbone ---
    base_model = TSception(
        nb_classes=1, Chans=Chans, Samples=Samples,
        sampling_rate=sampling_rate
    )
    # Extract up to Flatten (layer index 10) → 768-dim
    flatten_layer = None
    for layer in base_model.layers:
        if 'Flatten' in layer.__class__.__name__:
            flatten_layer = layer
            break
    
    backbone_output = flatten_layer.output if flatten_layer is not None else base_model.layers[10].output
    
    backbone = Model(
        inputs=base_model.input,
        outputs=backbone_output,
        name='tsception_backbone'
    )
    deep_features = backbone(raw_input)                # (batch, 768)
    d1 = Dense(deep_proj_dim, activation='relu', name='deep_proj')(deep_features)
    d1 = BatchNormalization(name='deep_bn')(d1)        # (batch, 128)

    # --- Stream 2: Handcrafted feature projection ---
    d2 = Dense(hc_proj_dim, activation='relu', name='hc_proj')(hc_input)
    d2 = BatchNormalization(name='hc_bn')(d2)          # (batch, 64)

    # --- Gated Fusion ---
    d1_reduced = Dense(hc_proj_dim, activation='relu', name='d1_reduce')(d1)  # (batch, 64)
    concat_for_gate = Concatenate(name='gate_input')([d1_reduced, d2])        # (batch, 128)
    gate = Dense(hc_proj_dim, activation='sigmoid', name='gate')(concat_for_gate)  # (batch, 64)

    # fused = gate ⊙ d1_reduced + (1-gate) ⊙ d2
    gated_deep = Multiply(name='gated_deep')([gate, d1_reduced])
    inv_gate = Lambda(lambda g: 1.0 - g, name='inv_gate')(gate)
    gated_hc = Multiply(name='gated_hc')([inv_gate, d2])
    fused = tf.keras.layers.Add(name='fused')([gated_deep, gated_hc])  # (batch, 64)

    # Concatenate fused representation with full deep projection for capacity
    combined = Concatenate(name='combined')([fused, d1])  # (batch, 192)

    # --- Shared representation ---
    shared = Dense(shared_dim, activation='relu', name='shared_dense')(combined)
    shared = BatchNormalization(name='shared_bn')(shared)
    shared = Dropout(dropout_rate, name='shared_dropout')(shared)

    # --- Task Heads ---
    dep_hidden = Dense(32, activation='relu', name='dep_hidden')(shared)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_hidden)

    str_hidden = Dense(32, activation='relu', name='str_hidden')(shared)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_hidden)

    model = Model(
        inputs=[raw_input, hc_input],
        outputs=[dep_out, str_out],
        name='DSFFN'
    )
    return model
