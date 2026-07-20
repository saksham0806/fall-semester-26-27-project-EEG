import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from ..proposed_model.eeg_models import EEGNet, TSception, EEGTransformer

def build_submodel_with_embedding(architecture, embed_dim=64, Chans=6, Samples=512):
    """
    Builds a sub-model that exposes both embedding and task predictions.

    Args:
        architecture: 'EEGNet', 'TSception', or 'EEGTransformer'
        embed_dim: dimensionality of the projection space for contrastive learning
        Chans: number of channels
        Samples: number of samples

    Returns:
        Model: Model with outputs [dep_out, str_out, embedding]
    """
    def get_target_layer_output(model, layer_class_name, default_idx):
        for layer in model.layers:
            if layer_class_name in layer.__class__.__name__:
                return layer.output
        return model.layers[default_idx].output

    if architecture == 'EEGNet':
        base = EEGNet(nb_classes=1, Chans=Chans, Samples=Samples)
        flatten_output = get_target_layer_output(base, 'Flatten', 13)
    elif architecture == 'TSception':
        base = TSception(nb_classes=1, Chans=Chans, Samples=Samples, sampling_rate=128)
        flatten_output = get_target_layer_output(base, 'Flatten', 10)
    elif architecture == 'EEGTransformer':
        base = EEGTransformer(nb_classes=1, Chans=Chans, Samples=Samples)
        flatten_output = get_target_layer_output(base, 'GlobalAveragePooling1D', -5)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Projection to shared embedding space
    proj = Dense(embed_dim, activation='relu',
                 name=f'{architecture}_proj')(flatten_output)
    embedding = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name=f'{architecture}_embedding'
    )(proj)

    # Task heads
    shared = Dense(32, activation='relu', name=f'{architecture}_shared')(proj)
    dep_out = Dense(1, activation='sigmoid', name=f'{architecture}_depression')(shared)
    str_out = Dense(1, activation='sigmoid', name=f'{architecture}_stress')(shared)

    full_model = Model(
        inputs=base.input,
        outputs=[dep_out, str_out, embedding],
        name=f'{architecture}_submodel'
    )
    return full_model

def build_meta_learner(num_submodels=3, num_datasets=4):
    """
    Meta-learner that combines predictions from all sub-models.

    Input:  [p_A_dep, p_A_str, p_B_dep, p_B_str, p_C_dep, p_C_str, ds_onehot]
    Output: [depression, stress]
    """
    # 6 prediction scores + 4 dataset one-hot = 10
    meta_input = Input(shape=(num_submodels * 2 + num_datasets,), name='meta_input')
    h = Dense(16, activation='relu', name='meta_hidden')(meta_input)
    dep_out = Dense(1, activation='sigmoid', name='depression')(h)
    str_out = Dense(1, activation='sigmoid', name='stress')(h)

    return Model(inputs=meta_input, outputs=[dep_out, str_out], name='MetaLearner')

def supervised_contrastive_loss(embeddings, labels, dataset_ids, temperature=0.07):
    """
    SupCon loss that pulls same-class embeddings together
    ACROSS datasets and pushes different classes apart.

    Args:
        embeddings: (batch, embed_dim) — L2-normalized
        labels:     (batch,) — 0 or 1 (task label)
        dataset_ids: (batch,) — 0, 1, 2, or 3
        temperature: float — scaling temperature

    Returns:
        scalar loss
    """
    # Normalize embeddings to unit sphere
    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    # Compute similarity matrix
    sim_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)  # (B, B)
    sim_matrix = sim_matrix / temperature

    # Create label match mask: positive pairs = same class
    labels = tf.cast(labels, tf.int32)
    label_match = tf.equal(
        tf.expand_dims(labels, 1),
        tf.expand_dims(labels, 0)
    )  # (B, B) boolean

    # Cross-dataset bonus: upweight positive pairs from DIFFERENT datasets
    ds_match = tf.equal(
        tf.expand_dims(dataset_ids, 1),
        tf.expand_dims(dataset_ids, 0)
    )  # (B, B) boolean
    cross_ds_weight = tf.where(ds_match, 1.0, 2.0)  # 2x weight for cross-dataset positives

    # Self-mask: exclude diagonal
    batch_size = tf.shape(embeddings)[0]
    self_mask = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))

    # Positive mask: same class AND not self
    pos_mask = tf.logical_and(label_match, self_mask)
    pos_mask_float = tf.cast(pos_mask, tf.float32) * cross_ds_weight

    # Numerator: exp(sim) for positive pairs
    exp_sim = tf.exp(sim_matrix) * tf.cast(self_mask, tf.float32)
    pos_sim = exp_sim * pos_mask_float

    # Denominator: sum of exp(sim) for all non-self pairs
    denom = tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-7

    # Loss: -log(sum_positives / sum_all)
    loss = -tf.math.log(
        tf.reduce_sum(pos_sim, axis=1) / tf.squeeze(denom) + 1e-7
    )

    # Only compute loss for samples with at least one positive pair
    has_pos = tf.cast(tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1) > 0, tf.float32)
    loss = tf.reduce_sum(loss * has_pos) / (tf.reduce_sum(has_pos) + 1e-7)

    return loss
