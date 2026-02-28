"""
models.py — Model definitions and training utilities.

Available models:
  1. build_custom_cnn()  — 3-block CNN trained from scratch
  2. build_mobilenet()   — MobileNetV2 with a custom classification head
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)

from src.config import (
    IMG_SIZE, NUM_CLASSES, LEARNING_RATE,
    CUSTOM_CNN_CHECKPOINT, MOBILENET_CHECKPOINT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CNN
# ─────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(
    input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """
    Build a 3-block CNN compiled and ready to train.

    Architecture
    ------------
    Block 1 : Conv(32)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
    Block 2 : Conv(64)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
    Block 3 : Conv(128) × 2 → BatchNorm → MaxPool → Dropout(0.25)
    Head    : Dense(512) → Dense(256) → Dense(num_classes, softmax)
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classification head
        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ], name="CustomCNN")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV2 Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenet(
    input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    fine_tune_at: int = 100,
) -> Sequential:
    """
    Build a MobileNetV2 transfer-learning model (base frozen, head trainable).

    Training is done in two phases:
      - Phase 1: train only the classification head (base frozen).
      - Phase 2: call unfreeze_mobilenet() to fine-tune the upper layers.

    Parameters
    ----------
    fine_tune_at : Layer index from which layers become trainable in Phase 2.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ], name="MobileNetV2_TL")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_mobilenet(model: Sequential, fine_tune_at: int = 100,
                       learning_rate: float = 1e-5) -> Sequential:
    """
    Unfreeze layers from *fine_tune_at* onwards and recompile for fine-tuning.

    Call this after Phase 1 training is complete to start Phase 2.
    """
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Fine-tuning: {sum(1 for l in base_model.layers if l.trainable)} "
          f"layers unfrozen (from layer {fine_tune_at}).")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks factory
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path: str, patience_es: int = 8,
                  patience_lr: int = 3) -> list:
    """Return a standard set of training callbacks (LR scheduler, early stopping, checkpoint)."""
    return [
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=patience_lr,
            verbose=1,
            min_lr=1e-7,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            verbose=1,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
