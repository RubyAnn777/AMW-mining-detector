# A collection of models explored, Aug / Sept 2025

# >> FILE OVERVIEW
# >> Recipe book of neural-network designs (Keras / TensorFlow) tried for mine detection.
# >> Production does NOT call these at inference: it loads the trained ensemble file
# >> models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5 (6 members, output (N, 6)).
# >> IN PRODUCTION: CNN800k (x5 members) and CNN2024 (x1, member v4.21 'CNN100ka').
# >> EXPLORED ONLY: ResNet18, ResNet50, MLP, MLP_with_targeted_dropout,
# >> LogisticRegression.
# >> Imported by core/model_evaluation.ipynb. Upstream Amazon Mining Watch file.
import tensorflow as tf

try:
    from .tf_darwin import apply_darwin_tf_compat
except ImportError:
    from tf_darwin import apply_darwin_tf_compat

# >> macOS (Darwin) TensorFlow compatibility patch, applied before keras is imported.
apply_darwin_tf_compat()

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
# >> Gotcha: this imported Keras ResNet50 is overwritten by the local def ResNet50 below,
# >> so the local function ends up calling itself (see ResNet50).
from tensorflow.keras.applications import ResNet50


# >> LAYER GLOSSARY (plain words):
# >> Conv2D = slides small 3x3 pattern detectors over the image; each 'filter' learns one
# >> pattern (edge, texture). 'same' padding keeps height/width.
# >> ReLU = keep positive values, set negatives to 0.
# >> BatchNormalization = rescales each layer's outputs to a steady range; trains faster.
# >> MaxPooling2D(2) = keep the max of each 2x2 block: halves height and width.
# >> Flatten = unroll the 3-D feature block into one long list of numbers.
# >> Dense = fully connected layer: every input number feeds every output unit.
# >> Dropout(0.3) = during training, randomly switch off 30% of units (less overfitting).
# >> Dense(1, sigmoid) = final score squashed into 0..1 = probability of 'mine'.
# 2024 v3 CNN ~ 100k params for 48px patch input.

# >> IN PRODUCTION (1 of 6 ensemble members: v4.21 'CNN100ka'). ~100k weights.
# >> Input (48, 48, 13) = 48x48 px chip, 13 Sentinel-2 bands. Output (N, 1) prob 0..1.
# >> Three blocks of 3 x Conv2D(32 filters, 3x3, ReLU), NO BatchNorm, each block ends
# >> with max-pool 2, 2, then 3. Shapes: 48 -> 24 -> 12 -> 4 px; flatten 4*4*32 = 512.
# >> Head: Dense 64 -> 64 -> 32 (each with Dropout 0.3) -> Dense 1 sigmoid.
def CNN2024(input_shape=(48,48,13)):
    """Construct the CNN used in 2024."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.MaxPooling2D(pool_size=(2)),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.MaxPooling2D(pool_size=(2)),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        # >> Pool size 3 here (not 2): 12x12 -> 4x4.
        layers.MaxPooling2D(pool_size=(3)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        # >> Final layer: one unit, sigmoid -> probability that the chip shows mining.
        layers.Dense(1, activation='sigmoid')])
    
    return model

# 800k param CNN

# >> IN PRODUCTION (5 of 6 ensemble members: v4.10b, 4.18d, 4.20g, 4.22b, 4.22c).
# >> Input (48, 48, 13). Output (N, 1) probability 0..1.
# >> 3 conv blocks of 32 / 64 / 128 filters; each = 3 x (Conv2D 3x3 ReLU + BatchNorm),
# >> then 2x2 max-pool. Shapes: 48 -> 24 -> 12 -> 6 px; flatten 6*6*128 = 4,608.
# >> Head: Dense 64 -> 64 -> 32 with Dropout 0.3 -> Dense 1 sigmoid.
# >> My count gives ~0.9M weights (name says 800k).
def CNN800k(input_shape=(48,48,13)):
    """Construct a CNN with ~800k parameters, given 48px patch input."""

    model = keras.Sequential([
        keras.Input(shape=input_shape),
    
        # Block 1
        # >> Block 1: 32 filters at 48x48; pool -> 24x24x32.
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Block 2
        # >> Block 2: 64 filters at 24x24; pool -> 12x12x64.
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Block 3
        # >> Block 3: 128 filters at 12x12; pool -> 6x6x128.
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # >> (6, 6, 128) -> (4608,) flat vector.
        layers.Flatten(),

        # Fully connected layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
    
        # >> Output: probability 0..1 of 'mine' for the chip.
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# ResNet-18, 11M params

# >> EXPLORED ONLY (ResNet helper). One Conv2D + BatchNorm + ReLU, with L2 weight
# >> decay 1e-4 (penalises large weights). Not used by ResNet18 below.
def conv_block(x, filters, kernel_size=3, stride=1, use_bias=False, weight_decay=1e-4):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# >> EXPLORED ONLY. ResNet 'residual block': two 3x3 convs whose output is ADDED to
# >> the block's input (the 'shortcut'), which lets very deep nets train.
# >> downsample = optional 1x1 conv so the shortcut matches a new size/filter count.
def residual_block(x, filters, stride=1, downsample=None, weight_decay=1e-4):
    shortcut = x

    # First conv
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    # Downsample shortcut if needed
    if downsample is not None:
        shortcut = downsample

    # Add + ReLU
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# >> EXPLORED ONLY. Stacks `blocks` residual blocks; the first may shrink via stride.
def make_layer(x, filters, blocks, stride=1, weight_decay=1e-4):
    # Downsample if stride != 1 or filter mismatch
    downsample = None
    input_filters = x.shape[-1]
    if stride != 1 or input_filters != filters:
        downsample = layers.Conv2D(
            filters, 1, strides=stride, padding='same', use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay))(x)
        downsample = layers.BatchNormalization()(downsample)

    x = residual_block(x, filters, stride=stride, downsample=downsample,
                       weight_decay=weight_decay)
    for _ in range(1, blocks):
        x = residual_block(x, filters, stride=1, weight_decay=weight_decay)
    return x

# >> EXPLORED ONLY. ResNet-18 (~11M weights) built from scratch for 13-band 48 px chips.
# >> Stem: 7x7 conv stride 2 + max-pool stride 2 (48 -> 24 -> 12 px), then 4 stages
# >> (64/128/256/512 filters; 12 -> 12 -> 6 -> 3 -> 2 px), global average pool,
# >> Dense 512 + Dropout 0.5, Dense 1 sigmoid. Not in the production ensemble.
def ResNet18(input_shape=(48,48,13), num_classes=1, weight_decay=1e-4):
    """Construct a ResNet-18 model."""
    inputs = layers.Input(shape=input_shape)

    # First conv adapted for 13 channels
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual layers
    x = make_layer(x, 64, blocks=2, stride=1, weight_decay=weight_decay)
    x = make_layer(x, 128, blocks=2, stride=2, weight_decay=weight_decay)
    x = make_layer(x, 256, blocks=2, stride=2, weight_decay=weight_decay)
    x = make_layer(x, 512, blocks=2, stride=2, weight_decay=weight_decay)

    # Global pooling + dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

# WIP ResNet 50. Adjusted for 13 input bands but the pretrained
# model still expects (224,224) rasters. 

# >> EXPLORED ONLY, WIP, and BROKEN as written: this def replaces the imported Keras
# >> ResNet50, so the call below recurses into itself with unexpected keyword
# >> arguments (include_top, weights...) -> TypeError. Also ImageNet weights expect
# >> 3-band 224 px input. Not in the production ensemble.
def ResNet50(input_shape=(48,48,13), num_classes=1, weight_decay=1e-4):
    """Construct a ResNet-50 pretrained on ImageNet, for input 13 bands."""
    inputs = layers.Input(shape=input_shape)

    # --- First conv replacement ---
    # Original ResNet50 expects 3 channels; we use 13 instead
    x = layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # --- Load ResNet50 without the top layers ---
    # Use include_top=False, no input layer needed since we manually handle it
    # >> Intended: Keras' pretrained ResNet50 (ImageNet). Actually calls this function.
    base = ResNet50(
        include_top=False,
        weights='imagenet',   # pretrained on ImageNet
        input_tensor=x,
        pooling=None
    )

    # --- Global Pooling + Classification Head ---
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model



# >> EXPLORED ONLY. Small fully connected net (MLP = multi-layer perceptron) on a
# >> feature vector, e.g. a 384-number SSL4EO ViT embedding (cls_only path, unused in
# >> production). Output (N, 1) sigmoid. Note: returned uncompiled despite docstring.
def MLP(
    input_dim: int = 384,
    hidden_layers: tuple = (64, 16),
    activation: str = "relu",
    output_activation: str = "sigmoid"
) -> tf.keras.Model:
    """
    Build a fully connected MLP for binary classification.

    Args:
        input_dim (int): Dimension of the input features (default 384).
        hidden_layers (tuple): Sizes of hidden layers, e.g. (64, 16).
        activation (str): Activation for hidden layers (default "relu").
        output_activation (str): Activation for output layer (default "sigmoid").

    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    # >> Stack Dense layers of the given sizes (default 64 then 16 units).
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))

    # Output layer: single neuron for binary classification
    model.add(layers.Dense(1, activation=output_activation))

    return model

# >> EXPLORED ONLY. MLP on [CLS || patch] ViT features (768 = 384 + 384); applies
# >> Dropout only to the CLS half. For the unused cls_patch embedding strategy.
def MLP_with_targeted_dropout(
    input_dim: int = 768,  # Total dim (384 CLS + 384 Patch)
    cls_input_dim: int = 384,
    hidden_layers: tuple = (32,),
    activation: str = "relu",
    cls_dropout_rate: float = 0.3,
    output_activation: str = "sigmoid"
) -> tf.keras.Model:
    """
    MLP on concatenated ViT features with dropout applied only to the CLS half.

    **Input layout** (must match training parquet / :func:`gee.embed_cls_patch.feature_column_names_cls_patch`):
    indices ``0 : cls_input_dim`` = class token (``cls0`` … ``cls{cls_input_dim-1}``),
    ``cls_input_dim :`` = one selected patch token (``spatial0`` …).

    Expects ``input_dim == 2 * cls_input_dim`` (e.g. 768 = 384 + 384 for ViT-S/16).
    """
    if input_dim != 2 * cls_input_dim:
        raise ValueError(
            f"input_dim ({input_dim}) must equal 2 * cls_input_dim ({2 * cls_input_dim}) "
            "for [CLS || patch] features from cls+patch embedding export"
        )

    # 1. Define Input
    inputs = layers.Input(shape=(input_dim,))

    # 2. Split the input into CLS and Patch tokens
    d = cls_input_dim
    # >> Split the (N, 768) input into CLS (N, 384) and patch token (N, 384).
    cls_token = layers.Lambda(lambda x, d=d: x[:, :d])(inputs)
    patch_token = layers.Lambda(lambda x, d=d: x[:, d:])(inputs)

    # 3. Apply Dropout ONLY to the CLS token
    if cls_dropout_rate > 0:
        cls_token = layers.Dropout(cls_dropout_rate)(cls_token)

    # 4. Concatenate them back together
    # >> Rejoin -> (N, 768).
    x = layers.Concatenate()([cls_token, patch_token])

    # 5. Hidden Layers
    for units in hidden_layers:
        x = layers.Dense(units, activation=activation)(x)

    # 6. Output Layer
    outputs = layers.Dense(1, activation=output_activation)(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# >> EXPLORED ONLY. Logistic regression = one Dense unit with sigmoid: a weighted sum
# >> of the 384 embedding numbers turned into a probability. Simplest baseline.
def LogisticRegression(
    input_dim: int = 384,
    output_activation: str = "sigmoid",
) -> tf.keras.Model:
    """
    Build a single-layer logistic regression model for binary classification.

    Args:
        input_dim (int): Dimension of the input features (default 384).
        output_activation (str): Activation for output layer (default "sigmoid").

    Returns:
        tf.keras.Model: Keras Sequential logistic regression model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(1, activation=output_activation),
        ]
    )
    return model
