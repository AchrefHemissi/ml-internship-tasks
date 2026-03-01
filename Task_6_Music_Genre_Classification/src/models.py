"""
models.py — Model factories, training helpers, and evaluation utilities
for both tabular (scikit-learn) and image-based (Keras/TensorFlow) approaches.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import PRIMARY_COLOR, SECONDARY_COLOR


# ──────────────────────────────────────────────────────────────
# Tabular models (scikit-learn)
# ──────────────────────────────────────────────────────────────

def get_tabular_models():
    """Return a dict of {name: unfitted sklearn estimator}."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "MLP Neural Net":      MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                             max_iter=500, random_state=42),
    }


def evaluate_classifier(model, X_train, X_test, y_train, y_test, label_encoder):
    """
    Fit a model, evaluate on test set, return (accuracy, train_time, y_pred, report).
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    return acc, elapsed, y_pred, report


def plot_confusion_matrix(y_true, y_pred, class_names, title,
                          cmap="Blues", normalize=False):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm  = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True label",  fontsize=12)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────
# CNN model (custom)
# ──────────────────────────────────────────────────────────────

def build_cnn(num_classes, img_size=(128, 128)):
    """
    4-block custom CNN for 128×128 Mel-spectrogram classification.

    Architecture:
      Block 1: Conv32 → Conv32 → BN → Pool → Dropout(0.25)
      Block 2: Conv64 → Conv64 → BN → Pool → Dropout(0.25)
      Block 3: Conv128 → Conv128 → BN → Pool → Dropout(0.25)
      Block 4: Conv256 → BN → Pool → Dropout(0.25)
      Head:    Dense512 → Dense256 → Softmax(num_classes)
    """
    from tensorflow.keras import layers, models as km, optimizers

    model = km.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(*img_size, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────
# Transfer Learning model (VGG16)
# ──────────────────────────────────────────────────────────────

def build_transfer_model(num_classes, img_size=(128, 128)):
    """
    VGG16-based transfer learning model.
    VGG16 base is frozen; a custom classification head is added on top.
    """
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras import layers, Model, optimizers

    base = VGG16(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
    for layer in base.layers:
        layer.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(checkpoint_path, patience=8):
    """Standard Keras callbacks: EarlyStopping, ReduceLR, ModelCheckpoint."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience // 2,
                          min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                        save_best_only=True, mode="max"),
    ]


def make_generators(spec_dir, img_size=(128, 128), batch_size=32):
    """
    Build Keras ImageDataGenerators for train, val (from train/), and test.
    Returns (train_gen, val_gen, test_gen).
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(spec_dir, "train"),
        target_size=img_size, batch_size=batch_size,
        class_mode="categorical", subset="training", shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        os.path.join(spec_dir, "train"),
        target_size=img_size, batch_size=batch_size,
        class_mode="categorical", subset="validation", shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(spec_dir, "test"),
        target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ──────────────────────────────────────────────────────────────
# Shared plotting utilities
# ──────────────────────────────────────────────────────────────

def plot_training_history(history, title):
    """Plot accuracy and loss training curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title(f"{title} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title(f"{title} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()
