"""Deep learning architectures for website fingerprinting: Var-CNN and DF."""

from typing import Dict, Any, Tuple

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Dense, Activation, Concatenate, Dropout, BatchNormalization,
        Conv1D, MaxPooling1D, Flatten, ELU, Input
    )
    from tensorflow.keras.optimizers import Adam, Adamax
except ImportError:
    from keras.models import Model
    from keras.layers import (
        Dense, Activation, Concatenate, Dropout, Flatten, ELU
    )
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam, Adamax
    from keras import Input

from src.models.layers import dilated_basic_1d, basic_1d, ResNet18


def build_var_cnn(config: Dict[str, Any], mixture_num: int = 0) -> Model:
    """
    Constructs the multi-modal Var-CNN architecture combining direction sequences,
    packet inter-arrival time sequences, and statistical metadata.
    """
    num_mon_sites = config['num_mon_sites']
    num_unmon_sites_test = config.get('num_unmon_sites_test', 0)
    num_unmon_sites_train = config.get('num_unmon_sites_train', 0)
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    seq_length = config.get('seq_length', 5000)
    mixture = config.get('mixture', [["dir", "metadata"], ["time", "metadata"]])
    sub_comb = mixture[mixture_num] if mixture_num < len(mixture) else mixture[0]

    use_dir = 'dir' in sub_comb
    use_time = 'time' in sub_comb
    use_metadata = 'metadata' in sub_comb

    dir_dilations = config.get('dir_dilations', True)
    time_dilations = config.get('time_dilations', True)

    input_tensors = []
    feature_tensors = []

    if use_dir:
        dir_input = Input(shape=(seq_length, 1), name='dir_input')
        block_fn = dilated_basic_1d if dir_dilations else basic_1d
        dir_features = ResNet18(dir_input, 'dir', block=block_fn)
        input_tensors.append(dir_input)
        feature_tensors.append(dir_features)

    if use_time:
        time_input = Input(shape=(seq_length, 1), name='time_input')
        block_fn = dilated_basic_1d if time_dilations else basic_1d
        time_features = ResNet18(time_input, 'time', block=block_fn)
        input_tensors.append(time_input)
        feature_tensors.append(time_features)

    if use_metadata:
        metadata_input = Input(shape=(7,), name='metadata_input')
        x = Dense(32)(metadata_input)
        x = BatchNormalization()(x)
        metadata_features = Activation('relu')(x)
        input_tensors.append(metadata_input)
        feature_tensors.append(metadata_features)

    if len(feature_tensors) == 1:
        combined = feature_tensors[0]
    else:
        combined = Concatenate()(feature_tensors)
        combined = Dense(config.get('dense_units', 1024))(combined)
        combined = BatchNormalization()(combined)
        combined = Activation('relu')(combined)
        combined = Dropout(config.get('dropout_rate', 0.5))(combined)

    output_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    model_output = Dense(units=output_classes, activation='softmax', name='model_output')(combined)

    model = Model(inputs=input_tensors, outputs=model_output)
    learning_rate = config.get('learning_rate', 0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy']
    )
    return model


def build_df_model(config: Dict[str, Any]) -> Model:
    """
    Constructs the Deep Fingerprinting (DF) 1D CNN baseline model.
    """
    num_mon_sites = config['num_mon_sites']
    num_unmon_sites_test = config.get('num_unmon_sites_test', 0)
    num_unmon_sites_train = config.get('num_unmon_sites_train', 0)
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train
    seq_length = config.get('seq_length', 5000)

    dir_input = Input(shape=(seq_length, 1), name='dir_input')

    # Block 1
    x = Conv1D(32, 8, strides=1, padding='same')(dir_input)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = Conv1D(32, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)

    # Block 2
    x = Conv1D(64, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)

    # Block 3
    x = Conv1D(128, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)

    # Block 4
    x = Conv1D(256, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)

    # Dense layers
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    output_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    model_output = Dense(units=output_classes, activation='softmax', name='model_output')(x)

    model = Model(inputs=dir_input, outputs=model_output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adamax(0.002),
        metrics=['accuracy']
    )
    return model


def build_model(config: Dict[str, Any], mixture_num: int = 0) -> Model:
    """
    Factory function to build either Var-CNN or DF model based on config.
    """
    model_name = config.get('model_name', 'var-cnn').lower()
    if model_name == 'var-cnn':
        return build_var_cnn(config, mixture_num=mixture_num)
    elif model_name == 'df':
        return build_df_model(config)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: 'var-cnn', 'df'")
