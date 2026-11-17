"""
app.py
--------
Gradio web application for the Car Body Type Classifier.

Features:
    - Upload / preview a car image
    - Predict the body type (Sedan / SUV / Hatchback / MPV/MUV)
    - Show confidence percentage
    - Show Top-3 predictions with probabilities
    - Show a short description of the predicted body type
    - Show a Grad-CAM heatmap of where the model focused (bonus)

Run:
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import os

import gradio as gr
from PIL import Image

from predict import predict_image, load_model, MODEL_PATH
from utils.image_utils import load_and_preprocess_image, make_gradcam_heatmap, overlay_gradcam_on_image


def classify_car_image(image):
    """
    Main callback wired to the Gradio interface.

    Parameters
    ----------
    image : PIL.Image.Image or None
        The uploaded image from the Gradio Image component.

    Returns
    -------
    Tuple matching the Gradio outputs:
        (prediction_label, confidence_text, top3_markdown, description_text, gradcam_image)
    """
    # --- Handle the "no image uploaded" case gracefully ---
    if image is None:
        return (
            "No image uploaded",
            "-",
            "Please upload a car image to see predictions.",
            "-",
            None,
        )

    try:
        result = predict_image(image, top_k=3)
    except ValueError as e:
        # Invalid / corrupted image
        return (f"Error: {e}", "-", "-", "-", None)
    except FileNotFoundError as e:
        # Model not trained yet
        return (f"Error: {e}", "-", "-", "-", None)
    except Exception as e:
        # Catch-all so the app never crashes on unexpected errors
        return (f"Unexpected error: {e}", "-", "-", "-", None)

    prediction_label = {name: conf / 100.0 for name, conf in result["top_k"]}
    confidence_text = f"{result['confidence']}%"

    top3_lines = [f"- **{name}**: {conf:.2f}%" for name, conf in result["top_k"]]
    top3_markdown = "\n".join(top3_lines)

    description_text = result["description"]

    # --- Bonus: Grad-CAM heatmap ---
    gradcam_image = None
    try:
        model = load_model()
        img_array = load_and_preprocess_image(image)
        heatmap = make_gradcam_heatmap(img_array, model)
        gradcam_image = overlay_gradcam_on_image(image, heatmap)
    except Exception as e:
        # Grad-CAM is a bonus visual feature - if it fails for any reason,
        # we still want the main prediction to be shown to the user.
        print(f"[WARNING] Grad-CAM generation failed: {e}")
        gradcam_image = None

    return prediction_label, confidence_text, top3_markdown, description_text, gradcam_image


def build_interface():
    """Construct and return the Gradio Blocks interface."""
    examples_list = []
    base_dir = os.path.dirname(__file__)
    example_candidates = [
        os.path.join(base_dir, "dataset", "test", "sedan", "download.jpg"),
        os.path.join(base_dir, "dataset", "test", "suv", "images (1).jpg"),
        os.path.join(base_dir, "dataset", "test", "hatchback", "images (1).jpg"),
        os.path.join(base_dir, "dataset", "test", "mpv", "images (1).jpg"),
    ]
    for p in example_candidates:
        if os.path.exists(p):
            examples_list.append([p])

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="Car Body Type Classifier") as demo:
        gr.Markdown(
            """
            # 🚗 Car Body Type Classification
            ### Transfer Learning with MobileNetV2

            Upload a car image and the model will classify it as
            **Sedan**, **SUV**, **Hatchback**, or **MPV/MUV**,
            along with a confidence score, Top-3 predictions breakdown, and Grad-CAM explainability heatmap.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Car Image")
                with gr.Row():
                    predict_btn = gr.Button("🔍 Classify Vehicle", variant="primary")
                    clear_btn = gr.ClearButton(components=[image_input], value="🗑️ Clear")
                gr.Markdown(
                    "💡 **Tip:** Use a clear, well-lit photo of the car "
                    "(preferably a side or 3/4 angle view) for best results."
                )

            with gr.Column(scale=1):
                prediction_output = gr.Label(label="Prediction & Confidence", num_top_classes=3)
                confidence_output = gr.Textbox(label="Top Class Confidence", interactive=False)
                top3_output = gr.Markdown(label="Top-3 Predictions Breakdown")
                description_output = gr.Textbox(label="Body Type Description", interactive=False, lines=3)

        with gr.Row():
            gradcam_output = gr.Image(type="pil", label="Grad-CAM Heatmap (Model Visual Attention)")

        if examples_list:
            gr.Examples(
                examples=examples_list,
                inputs=image_input,
                outputs=[prediction_output, confidence_output, top3_output, description_output, gradcam_output],
                fn=classify_car_image,
                cache_examples=False,
                label="Sample Test Images (Click to Test)",
            )

        predict_btn.click(
            fn=classify_car_image,
            inputs=image_input,
            outputs=[prediction_output, confidence_output, top3_output, description_output, gradcam_output],
        )

        # Also predict automatically when an image is uploaded
        image_input.change(
            fn=classify_car_image,
            inputs=image_input,
            outputs=[prediction_output, confidence_output, top3_output, description_output, gradcam_output],
        )

        gr.Markdown(
            """
            ---
            Built with **TensorFlow / Keras** (MobileNetV2 Transfer Learning) and **Gradio**.
            """
        )

    return demo, theme


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(
            f"[WARNING] No trained model found at '{MODEL_PATH}'.\n"
            f"The app will still launch, but predictions will fail until "
            f"you run 'python train.py' to train and save a model."
        )

    demo, theme = build_interface()
    demo.launch(theme=theme, inbrowser=True)
