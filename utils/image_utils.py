"""
image_utils.py
----------------
Reusable image helper functions shared by train.py, predict.py and app.py.

Includes:
    - load_and_preprocess_image : load a PIL image / path into a model-ready array
    - is_valid_image             : sanity check before trying to run prediction
    - make_gradcam_heatmap       : Grad-CAM heatmap generation (bonus feature)
    - overlay_gradcam_on_image   : blend the heatmap on top of the original image
"""

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMG_SIZE = (224, 224)  # MobileNetV2's expected input size


def is_valid_image(file_path_or_pil) -> bool:
    """
    Check whether the given input is a readable image.
    Accepts either a file path (str) or an already-loaded PIL Image.
    Returns True if valid, False otherwise. Never raises.

    IMPORTANT: PIL's Image.verify() permanently invalidates the image object
    it's called on (closes its internal file pointer), so it can never be
    used again afterwards. To avoid breaking a PIL.Image the caller still
    needs (e.g. the object handed to us by Gradio), we always verify a
    throwaway COPY rather than the original object/file handle.
    """
    try:
        if isinstance(file_path_or_pil, Image.Image):
            file_path_or_pil.copy().verify()
            return True
        with Image.open(file_path_or_pil) as img:
            img.copy().verify()
        return True
    except (UnidentifiedImageError, FileNotFoundError, OSError, AttributeError, ValueError):
        return False


def load_and_preprocess_image(image, target_size=IMG_SIZE):
    """
    Load an image (path or PIL.Image) and turn it into a batch of shape
    (1, 224, 224, 3), scaled to [0, 1], ready to feed into the model.

    Parameters
    ----------
    image : str | PIL.Image.Image
        Path to an image file, or an already-loaded PIL image (e.g. from Gradio).
    target_size : tuple
        Target (width, height) for resizing. Defaults to MobileNetV2's 224x224.

    Returns
    -------
    np.ndarray of shape (1, 224, 224, 3), dtype float32
    """
    if isinstance(image, str):
        pil_img = Image.open(image)
    else:
        pil_img = image

    # Ensure 3 channels (handles grayscale / RGBA uploads gracefully)
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)

    img_array = np.array(pil_img, dtype=np.float32)
    img_array = img_array / 255.0  # Normalise to [0, 1], matches training pipeline
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    """
    Generate a Grad-CAM heatmap for the given preprocessed image array.

    Parameters
    ----------
    img_array : np.ndarray, shape (1, 224, 224, 3)
        Preprocessed input image (same format used for prediction).
    model : tf.keras.Model
        The trained classification model.
    last_conv_layer_name : str
        Name of the last convolutional layer of the MobileNetV2 backbone.
        "Conv_1" is the last conv layer name in Keras' MobileNetV2 implementation.
    pred_index : int or None
        Class index to explain. If None, uses the model's top predicted class.

    Returns
    -------
    heatmap : np.ndarray, values in [0, 1], shape (H, W)
    """
    # Locate the base MobileNetV2 model nested inside our Sequential/Functional model
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break

    target_model = base_model if base_model is not None else model

    try:
        last_conv_layer = target_model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback: search for the last Conv2D layer in the target model
        last_conv_layer = None
        for layer in reversed(target_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    # Because the base model is nested inside our outer model (Keras 3 treats
    # it as a single functional sub-graph), we can't directly trace a path
    # from the OUTER model's input to an INNER layer's output in one go.
    # Instead we:
    #   1. Build a small model: base_model.input -> (last_conv_layer.output, base_model.output)
    #   2. Manually replay the remaining "head" layers (GAP, Dense, Dropout, Dense)
    #      on top of base_model's output inside a GradientTape, so gradients
    #      can flow back to the conv feature map.
    grad_model = tf.keras.models.Model(
        inputs=target_model.input,
        outputs=[last_conv_layer.output, target_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)
        tape.watch(conv_outputs)

        x = base_output
        if base_model is not None:
            # Replay every layer of the outer model that comes AFTER the base model
            found_base = False
            for layer in model.layers:
                if layer is base_model:
                    found_base = True
                    continue
                if not found_base or isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam_on_image(pil_image, heatmap, alpha=0.4, target_size=IMG_SIZE):
    """
    Overlay a Grad-CAM heatmap on top of the original PIL image.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The original (unprocessed) image.
    heatmap : np.ndarray
        Heatmap values in [0, 1] as returned by make_gradcam_heatmap.
    alpha : float
        Blend strength of the heatmap overlay.
    target_size : tuple
        Size to resize both the base image and heatmap to.

    Returns
    -------
    PIL.Image.Image  — RGB image with the heatmap overlaid.
    """
    import matplotlib.cm as cm

    base_img = pil_image.convert("RGB").resize(target_size)
    base_arr = np.array(base_img)

    # Resize heatmap to match image size
    heatmap_resized = np.array(Image.fromarray(np.uint8(heatmap * 255)).resize(target_size))

    # Apply a colormap (jet) to the heatmap
    try:
        import matplotlib
        jet = matplotlib.colormaps["jet"]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Blend original image and heatmap
    overlaid = jet_heatmap * alpha + base_arr * (1 - alpha)
    overlaid = np.uint8(overlaid)

    return Image.fromarray(overlaid)
