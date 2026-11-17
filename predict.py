"""
predict.py
------------
Loads the trained model from saved_model/ and provides functions to run
predictions on a single image. Can be used:
    - as a module, imported by app.py
    - directly from the command line:  python predict.py path/to/car.jpg
"""

import os
import sys
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.class_names import CLASS_NAMES, get_display_name, get_description
from utils.image_utils import load_and_preprocess_image, is_valid_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "car_body_classifier.keras")

_model = None  # module-level cache so we only load the model once


def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load and cache the trained Keras model.
    Raises a clear error if the model file doesn't exist yet (i.e. training
    hasn't been run).
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'.\n"
            f"Please run 'python train.py' first to train and save the model."
        )

    print(f"[INFO] Loading model from {model_path} ...")
    _model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return _model


def predict_image(image, top_k: int = 3):
    """
    Run inference on a single image and return structured prediction results.

    Parameters
    ----------
    image : str | PIL.Image.Image
        Path to an image file, or a PIL image (e.g. from Gradio's upload widget).
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict with keys:
        "predicted_class"   : str  (raw class name, e.g. "suv")
        "display_name"      : str  (friendly name, e.g. "SUV")
        "confidence"        : float (0-100, confidence of top prediction)
        "top_k"             : list of (display_name, confidence) tuples, sorted desc
        "description"       : str  (short description of the predicted class)

    Raises
    ------
    ValueError if the image is invalid / unreadable.
    """
    if not is_valid_image(image):
        raise ValueError(
            "The uploaded file could not be read as a valid image. "
            "Please upload a JPG or PNG image of a car."
        )

    model = load_model()

    img_array = load_and_preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)

    # Sort predictions in descending order of probability
    sorted_indices = np.argsort(predictions)[::-1]

    top_k = min(top_k, len(CLASS_NAMES))
    top_predictions = []
    for idx in sorted_indices[:top_k]:
        class_name = CLASS_NAMES[idx]
        confidence = float(predictions[idx]) * 100
        top_predictions.append((get_display_name(class_name), round(confidence, 2)))

    best_idx = int(sorted_indices[0])
    best_class = CLASS_NAMES[best_idx]

    result = {
        "predicted_class": best_class,
        "display_name": get_display_name(best_class),
        "confidence": round(float(predictions[best_idx]) * 100, 2),
        "top_k": top_predictions,
        "description": get_description(best_class),
    }
    return result


def _print_result(result: dict):
    """Pretty-print a prediction result to the console."""
    print("\nPrediction:")
    print(f"  {result['display_name']}")
    print("\nConfidence:")
    print(f"  {result['confidence']}%")
    print("\nTop Predictions:")
    for name, conf in result["top_k"]:
        print(f"  {name}: {conf}%")
    print("\nDescription:")
    print(f"  {result['description']}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    try:
        result = predict_image(image_path)
        _print_result(result)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
