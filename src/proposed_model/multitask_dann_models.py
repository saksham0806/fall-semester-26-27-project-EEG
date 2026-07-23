import tensorflow as tf
from tensorflow.keras import layers, models

from .eeg_models import TransformerEncoderBlock

def masked_binary_crossentropy(y_true, y_pred):
    """
    Computes binary cross-entropy, masking out elements where y_true is -1.
    Useful for multi-task training where some datasets lack specific task labels.
    
    Args:
        y_true: Ground truth targets, shape (batch_size, 1) or (batch_size,). Can contain -1.
        y_pred: Predicted probabilities, shape (batch_size, 1) or (batch_size,).
        
    Returns:
        loss: Scalar mean loss computed only over valid (not -1) samples.
    """
    # Ensure consistent datatypes
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Create mask for valid labels (i.e. labels that are not -1)
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    
    # Replace -1 with 0 in ground truth to avoid negative value errors in binary_crossentropy
    y_true_clean = tf.where(tf.equal(y_true, -1.0), tf.zeros_like(y_true), y_true)
    
    # Calculate binary crossentropy per sample
    bce = tf.keras.losses.binary_crossentropy(y_true_clean, y_pred)
    
    # Force shapes to align for element-wise multiplication
    bce = tf.reshape(bce, tf.shape(mask))
    
    # Calculate sum of mask to compute mean over valid elements
    sum_mask = tf.reduce_sum(mask)
    
    # Avoid division by zero if all elements in the batch are masked out
    loss = tf.reduce_sum(bce * mask) / (sum_mask + 1e-8)
    return loss

@tf.custom_gradient
def gradient_reversal(x, alpha=1.0):
    """
    Custom gradient reversal function.
    In the forward pass, acts as identity mapping.
    In the backward pass, reverses and scales the incoming gradient.
    """
    def grad(dy):
        # Reverse the gradient and scale by alpha
        return -dy * alpha, None
    return tf.identity(x), grad

class GradientReversalLayer(layers.Layer):
    """
    Gradient Reversal Layer (GRL) for Domain Adversarial training.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, x):
        return gradient_reversal(x, self.alpha)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config

def build_multitask_eeg_model(backbone_type='TSception', Chans=6, Samples=512, sampling_rate=128):
    """
    Build a multi-task EEG model with a shared backbone and two distinct output heads.
    Output 1: Depression detection probability.
    Output 2: Active cognitive stress probability.
    
    Compiled with custom masked loss.
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')
    
    # Build shared feature extractor backbone
    if backbone_type == 'TSception':
        scales = [int(sampling_rate * 0.5), int(sampling_rate * 0.25), int(sampling_rate * 0.125)]
        t_layers = []
        for scale in scales:
            x = layers.Conv2D(9, (1, scale), padding='same', activation='relu')(input_layer)
            t_layers.append(x)
        x = layers.Concatenate(axis=-1)(t_layers)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(6, (Chans, 1), padding='valid', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.AveragePooling2D((1, 4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Flatten()(x)
        features = layers.Dense(128, activation='relu', name='shared_features')(x)
    else:
        # Fallback to standard 1D CNN encoder
        x = layers.Reshape((Chans, Samples))(input_layer)
        x = layers.Permute((2, 1))(x)
        x = layers.Conv1D(32, 10, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 10, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        features = layers.Dense(128, activation='relu', name='shared_features')(x)
        
    # Multi-task heads
    depression_out = layers.Dense(1, activation='sigmoid', name='depression')(features)
    stress_out = layers.Dense(1, activation='sigmoid', name='stress')(features)
    
    model = models.Model(inputs=input_layer, outputs=[depression_out, stress_out], name='multitask_eeg_model')
    
    # Compile model with masked losses
    model.compile(
        optimizer='adam',
        loss={
            'depression': masked_binary_crossentropy,
            'stress': masked_binary_crossentropy
        },
        metrics={
            'depression': 'accuracy',
            'stress': 'accuracy'
        }
    )
    return model

def build_dann_eeg_model(backbone_type='TSception', nb_classes=1, nb_domains=4, Chans=6, Samples=512, sampling_rate=128, alpha=1.0):
    """
    Build a Domain Adversarial Neural Network (DANN) for EEG domain adaptation.
    Output 1: Main task classifier (e.g. stress/depression).
    Output 2: Domain classifier (dataset origin index) connected via Gradient Reversal Layer.

    `alpha` is fixed at construction. Standard DANN ramps it 0 -> 1 during training; sweeping
    fixed values answers the same question without the extra machinery.
    # ponytail: fixed alpha, ramp 0->1 via tf.Variable + callback if the sweep shows alpha matters
    """
    input_layer = layers.Input(shape=(Chans, Samples, 1), name='input')

    # Build shared feature extractor backbone
    # NOTE: this TSception is an inline copy that has diverged from eeg_models.TSception.
    # Left as-is deliberately — run_models.py depends on its current behaviour.
    if backbone_type == 'TSception':
        scales = [int(sampling_rate * 0.5), int(sampling_rate * 0.25), int(sampling_rate * 0.125)]
        t_layers = []
        for scale in scales:
            x = layers.Conv2D(9, (1, scale), padding='same', activation='relu')(input_layer)
            t_layers.append(x)
        x = layers.Concatenate(axis=-1)(t_layers)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(6, (Chans, 1), padding='valid', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.AveragePooling2D((1, 4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Flatten()(x)
        features = layers.Dense(128, activation='relu', name='shared_features')(x)
    elif backbone_type == 'EEGTransformer':
        # Same tokenisation as eeg_models.EEGTransformer, truncated before its head so the
        # task/domain wiring below is shared with the other backbones.
        patch_size, embed_dim, num_heads, ff_dim, num_layers = 16, 64, 4, 128, 2
        x = layers.Reshape((Chans, Samples))(input_layer)
        x = layers.Permute((2, 1))(x)
        x = layers.Conv1D(embed_dim, patch_size, strides=patch_size,
                          padding='valid', activation='relu')(x)
        seq_len = Samples // patch_size
        x = x + layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(
            tf.range(start=0, limit=seq_len, delta=1))
        for _ in range(num_layers):
            x = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads,
                                        ff_dim=ff_dim, rate=0.5)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        features = layers.Dense(128, activation='relu', name='shared_features')(x)
    else:
        # Fallback to standard 1D CNN encoder
        x = layers.Reshape((Chans, Samples))(input_layer)
        x = layers.Permute((2, 1))(x)
        x = layers.Conv1D(32, 10, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 10, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        features = layers.Dense(128, activation='relu', name='shared_features')(x)
        
    # Main Task Classifier Branch
    if nb_classes == 1:
        task_out = layers.Dense(1, activation='sigmoid', name='task')(features)
        task_loss = 'binary_crossentropy'
    else:
        task_out = layers.Dense(nb_classes, activation='softmax', name='task')(features)
        task_loss = 'sparse_categorical_crossentropy'
        
    # Domain Classifier Branch (with GRL)
    grl_features = GradientReversalLayer(alpha=alpha)(features)
    domain_out = layers.Dense(nb_domains, activation='softmax', name='domain')(grl_features)
    
    model = models.Model(inputs=input_layer, outputs=[task_out, domain_out], name='dann_eeg_model')
    
    # Compile DANN
    model.compile(
        optimizer='adam',
        loss={
            'task': task_loss,
            'domain': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'task': 1.0,
            'domain': 1.0
        },
        metrics={
            'task': 'accuracy',
            'domain': 'accuracy'
        }
    )
    return model
