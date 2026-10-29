"""ResNet 1D building blocks and layer definitions."""

try:
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Activation, ZeroPadding1D,
        GlobalAveragePooling1D, Add, BatchNormalization
    )
except ImportError:
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.layers import Activation, ZeroPadding1D, GlobalAveragePooling1D, Add
    from keras.layers.normalization import BatchNormalization

PARAMETERS = {'kernel_initializer': 'he_normal'}


def dilated_basic_1d(
    filters: int,
    suffix: str,
    stage: int = 0,
    block: int = 0,
    kernel_size: int = 3,
    numerical_name: bool = False,
    stride: int = None,
    dilations: tuple = (1, 1)
):
    """
    A 1D basic residual block with dilated causal convolutions.
    """
    if stride is None:
        stride = 1 if (block != 0 or stage == 0) else 2

    block_char = f'b{block}' if (block > 0 and numerical_name) else chr(ord('a') + block)
    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(
            filters, kernel_size, padding='causal', strides=stride,
            dilation_rate=dilations[0], use_bias=False,
            name=f'res{stage_char}{block_char}_branch2a_{suffix}', **PARAMETERS
        )(x)
        y = BatchNormalization(
            epsilon=1e-5, name=f'bn{stage_char}{block_char}_branch2a_{suffix}'
        )(y)
        y = Activation('relu', name=f'res{stage_char}{block_char}_branch2a_relu_{suffix}')(y)

        y = Conv1D(
            filters, kernel_size, padding='causal', use_bias=False,
            dilation_rate=dilations[1],
            name=f'res{stage_char}{block_char}_branch2b_{suffix}', **PARAMETERS
        )(y)
        y = BatchNormalization(
            epsilon=1e-5, name=f'bn{stage_char}{block_char}_branch2b_{suffix}'
        )(y)

        if block == 0:
            shortcut = Conv1D(
                filters, 1, strides=stride, use_bias=False,
                name=f'res{stage_char}{block_char}_branch1_{suffix}', **PARAMETERS
            )(x)
            shortcut = BatchNormalization(
                epsilon=1e-5, name=f'bn{stage_char}{block_char}_branch1_{suffix}'
            )(shortcut)
        else:
            shortcut = x

        y = Add(name=f'res{stage_char}{block_char}_{suffix}')([y, shortcut])
        y = Activation('relu', name=f'res{stage_char}{block_char}_relu_{suffix}')(y)
        return y

    return f


def basic_1d(
    filters: int,
    suffix: str,
    stage: int = 0,
    block: int = 0,
    kernel_size: int = 3,
    numerical_name: bool = False,
    stride: int = None,
    dilations: tuple = (1, 1)
):
    """
    A 1D basic residual block without dilations.
    """
    if stride is None:
        stride = 1 if (block != 0 or stage == 0) else 2

    dilations = (1, 1)
    block_char = f'b{block}' if (block > 0 and numerical_name) else chr(ord('a') + block)
    stage_char = str(stage + 2)

    def f(x):
        y = Conv1D(
            filters, kernel_size, padding='same', strides=stride,
            dilation_rate=dilations[0], use_bias=False,
            name=f'res{stage_char}{block_char}_branch2a_{suffix}', **PARAMETERS
        )(x)
        y = BatchNormalization(
            epsilon=1e-5, name=f'bn{stage_char}{block_char}_branch2a_{suffix}'
        )(y)
        y = Activation('relu', name=f'res{stage_char}{block_char}_branch2a_relu_{suffix}')(y)

        y = Conv1D(
            filters, kernel_size, padding='same', use_bias=False,
            dilation_rate=dilations[1],
            name=f'res{stage_char}{block_char}_branch2b_{suffix}', **PARAMETERS
        )(y)
        y = BatchNormalization(
            epsilon=1e-5, name=f'bn{stage_char}{block_char}_branch2b_{suffix}'
        )(y)

        if block == 0:
            shortcut = Conv1D(
                filters, 1, strides=stride, use_bias=False,
                name=f'res{stage_char}{block_char}_branch1_{suffix}', **PARAMETERS
            )(x)
            shortcut = BatchNormalization(
                epsilon=1e-5, name=f'bn{stage_char}{block_char}_branch1_{suffix}'
            )(shortcut)
        else:
            shortcut = x

        y = Add(name=f'res{stage_char}{block_char}_{suffix}')([y, shortcut])
        y = Activation('relu', name=f'res{stage_char}{block_char}_relu_{suffix}')(y)
        return y

    return f


def ResNet18(inputs, suffix: str, blocks=None, block=None, numerical_names=None):
    """
    Constructs a 1D ResNet-18 feature extraction backbone.
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]
    if block is None:
        block = dilated_basic_1d
    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = ZeroPadding1D(padding=3, name=f'padding_conv1_{suffix}')(inputs)
    x = Conv1D(64, 7, strides=2, use_bias=False, name=f'conv1_{suffix}')(x)
    x = BatchNormalization(epsilon=1e-5, name=f'bn_conv1_{suffix}')(x)
    x = Activation('relu', name=f'conv1_relu_{suffix}')(x)
    x = MaxPooling1D(3, strides=2, padding='same', name=f'pool1_{suffix}')(x)

    features = 64
    for stage_id, iterations in enumerate(blocks):
        x = block(features, suffix, stage_id, 0, dilations=(1, 2), numerical_name=False)(x)
        for block_id in range(1, iterations):
            x = block(
                features, suffix, stage_id, block_id, dilations=(4, 8),
                numerical_name=(block_id > 0 and numerical_names[stage_id])
            )(x)
        features *= 2

    x = GlobalAveragePooling1D(name=f'pool5_{suffix}')(x)
    return x
