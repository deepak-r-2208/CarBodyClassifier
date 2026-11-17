"""
train.py
----------
Trains a Car Body Type classifier (Sedan / SUV / Hatchback / MPV) using
Transfer Learning on top of MobileNetV2.

Pipeline:
    1. Load dataset from dataset/train, dataset/validation, dataset/test
    2. Apply data augmentation on the training set
    3. Build MobileNetV2 (frozen base) + custom classification head
    4. Train with EarlyStopping + ModelCheckpoint callbacks
    5. Plot & save accuracy/loss graphs
    6. Evaluate on the test set and save a confusion matrix
    7. Save the final trained model to saved_model/

Run:
    python train.py
"""

import os
import json

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so this also works on headless servers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from utils.class_names import CLASS_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT_LAYER = 100  # unfreeze from this layer index onwards during fine-tuning

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
BEST_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_checkpoint.keras")
FINAL_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "car_body_classifier.keras")
CLASS_INDICES_PATH = os.path.join(SAVED_MODEL_DIR, "class_indices.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)


def check_dataset_exists():
    """Make sure the dataset folders actually contain images before training."""
    for split_dir, split_name in [(TRAIN_DIR, "train"), (VAL_DIR, "validation"), (TEST_DIR, "test")]:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Missing dataset folder: {split_dir}\n"
                f"Please create dataset/{split_name}/<class_name>/ folders with images."
            )
        class_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        if len(class_folders) == 0:
            raise FileNotFoundError(f"No class sub-folders found inside {split_dir}")
        for cls in class_folders:
            cls_path = os.path.join(split_dir, cls)
            n_images = len([f for f in os.listdir(cls_path) if not f.startswith(".")])
            if n_images == 0:
                print(f"[WARNING] '{cls_path}' has no images yet. Add some before training.")


def build_data_generators():
    """
    Build ImageDataGenerators:
        - Training set gets augmentation (flip, rotation, zoom, etc.)
        - Validation/Test sets are only rescaled (no augmentation) to give a
          fair, unbiased evaluation.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build the transfer-learning model:
        MobileNetV2 (frozen, imagenet weights, no top)
          -> GlobalAveragePooling2D
          -> Dense(128, relu) + Dropout
          -> Dense(num_classes, softmax)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze the base model initially

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="car_body_classifier")
    return model, base_model


def get_callbacks():
    """EarlyStopping + ModelCheckpoint so we always keep the best-performing weights."""
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_CHECKPOINT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
    return [early_stop, checkpoint, reduce_lr]


def plot_training_history(history_1, history_2=None, save_path=os.path.join(MODEL_DIR, "training_history.png")):
    """Plot & save accuracy/loss curves. If fine-tuning history is given, it is appended."""
    acc = history_1.history["accuracy"]
    val_acc = history_1.history["val_accuracy"]
    loss = history_1.history["loss"]
    val_loss = history_1.history["val_loss"]

    if history_2 is not None:
        acc += history_2.history["accuracy"]
        val_acc += history_2.history["val_accuracy"]
        loss += history_2.history["loss"]
        val_loss += history_2.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Training history graph saved to {save_path}")


def evaluate_and_plot_confusion_matrix(model, test_generator):
    """Evaluate the model on the test set and save a confusion matrix heatmap."""
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("[INFO] Classification Report:\n", report)

    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}")


def main():
    print("[INFO] Checking dataset...")
    check_dataset_exists()

    print("[INFO] Building data generators...")
    train_gen, val_gen, test_gen = build_data_generators()

    print(f"[INFO] Classes found: {train_gen.class_indices}")
    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print("[INFO] Building model (MobileNetV2 base frozen)...")
    model, base_model = build_model(num_classes=len(CLASS_NAMES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("[INFO] Stage 1: Training classification head (base frozen)...")
    history_1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=INITIAL_EPOCHS,
        callbacks=get_callbacks(),
    )

    # --------------------------------------------------------------
    # Stage 2 (optional but recommended): Fine-tuning
    # Unfreeze the top layers of MobileNetV2 and train with a very
    # small learning rate to gently adapt pretrained features.
    # --------------------------------------------------------------
    print("[INFO] Stage 2: Fine-tuning top layers of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
    )

    print("[INFO] Plotting training history...")
    plot_training_history(history_1, history_2)

    print("[INFO] Evaluating on test set...")
    evaluate_and_plot_confusion_matrix(model, test_gen)

    print(f"[INFO] Saving final model to {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)
    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
