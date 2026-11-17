# 🚗 Car Body Type Classification using Transfer Learning

A beginner-friendly Deep Learning project that classifies an uploaded car image
into one of four body types — **Sedan**, **SUV**, **Hatchback**, or **MPV/MUV**
— using **Transfer Learning with MobileNetV2** and a clean **Gradio** web
interface. Built to be a solid college major project and resume piece.

---

## 📌 Project Overview

- **Task:** Multi-class image classification (4 classes)
- **Approach:** Transfer Learning — a pretrained **MobileNetV2** (ImageNet
  weights) is used as a frozen feature extractor, with a custom classification
  head trained on top. The top layers of MobileNetV2 are later fine-tuned with
  a very small learning rate for a further accuracy boost.
- **Frontend:** [Gradio](https://gradio.app) web app — upload an image, get an
  instant prediction with confidence score, Top-3 predictions, a short
  description of the predicted body type, and a Grad-CAM heatmap showing where
  the model focused.
- **Frameworks:** TensorFlow / Keras, Python 3.11+

### Why Transfer Learning?

Training a CNN from scratch needs a huge dataset and a lot of compute. By
reusing MobileNetV2's features (already learned from 1.4M+ ImageNet images),
we get strong accuracy even with a relatively small custom dataset of car
images, and training finishes in minutes instead of hours.

---

## 📂 Folder Structure

```
CarBodyClassifier/
│── app.py                  # Gradio web application (frontend)
│── train.py                # Model training script (transfer learning)
│── predict.py               # Loads saved model & runs predictions
│── requirements.txt         # Python dependencies
│── README.md                 # This file
│
│── model/                   # Training artifacts (auto-generated)
│     training_history.png    # Accuracy / loss curves
│     confusion_matrix.png    # Confusion matrix on test set
│     classification_report.txt
│     best_checkpoint.keras   # Best checkpoint saved during training
│
│── dataset/                 # Your car images go here (see below)
│     train/
│       sedan/  suv/  hatchback/  mpv/
│     validation/
│       sedan/  suv/  hatchback/  mpv/
│     test/
│       sedan/  suv/  hatchback/  mpv/
│
│── utils/
│     image_utils.py         # Preprocessing + Grad-CAM helper functions
│     class_names.py         # Class labels & body-type descriptions
│
│── static/                  # Optional static assets for the app
│── saved_model/              # Final trained model (auto-generated)
│     car_body_classifier.keras
│     class_indices.json
│── screenshots/              # App screenshots for this README (add your own)
```

---

## ⚙️ Installation

1. **Clone / download this project folder.**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > Requires Python 3.11+. If you have a GPU, make sure you install a
   > TensorFlow build with GPU support for faster training (CPU also works
   > fine for a dataset of this size — training just takes a bit longer).

---

## 🗂️ Dataset Format

Organise your images exactly like this (folder names must be lowercase and
match the classes below):

```
dataset/
  train/
    sedan/
    suv/
    hatchback/
    mpv/
  validation/
    sedan/
    suv/
    hatchback/
    mpv/
  test/
    sedan/
    suv/
    hatchback/
    mpv/
```

- Accepted formats: `.jpg`, `.jpeg`, `.png`
- Recommended: **150–200+ images per class** for training, with validation
  and test sets around 15–20% the size of the training set.
- Use varied angles, colors, lighting and backgrounds for better generalisation.

You can source images manually, scrape them (respecting copyright/usage
terms), or use an existing car-classification dataset and re-sort it into
these four folders.

---

## 🏋️ Training the Model

Once your `dataset/` folder is populated, run:

```bash
python train.py
```

What happens during training:

1. Loads training/validation/test data using
   `ImageDataGenerator.flow_from_directory` (automatic labeling from folder names).
2. Applies **data augmentation** (rotation, zoom, shifts, flips, brightness)
   to the training set only.
3. Builds **MobileNetV2** with `include_top=False` and **freezes** its
   weights, then adds:
   - `GlobalAveragePooling2D`
   - `Dense(128, relu)` + `Dropout(0.3)`
   - `Dense(4, softmax)`
4. Trains the classification head first (base frozen), using:
   - **EarlyStopping** (stops if validation loss stops improving)
   - **ModelCheckpoint** (saves the best-performing weights)
   - **ReduceLROnPlateau** (lowers the learning rate when training plateaus)
5. **Fine-tunes** the top layers of MobileNetV2 with a very small learning
   rate for a few more epochs.
6. Saves:
   - `model/training_history.png` — accuracy & loss curves
   - `model/confusion_matrix.png` — confusion matrix on the test set
   - `model/classification_report.txt` — precision/recall/F1 per class
   - `saved_model/car_body_classifier.keras` — the final trained model
   - `saved_model/class_indices.json` — class name ↔ index mapping

Training time depends on dataset size and hardware, but with MobileNetV2's
lightweight architecture it's usually fast even on CPU.

---

## 🔮 Running Predictions

### Option 1 — Command line

```bash
python predict.py path/to/your/car_image.jpg
```

Example output:
```
Prediction:
  SUV

Confidence:
  96.82%

Top Predictions:
  SUV — 96.82%
  MPV/MUV — 2.11%
  Sedan — 1.07%

Description:
  SUVs (Sport Utility Vehicles) usually have higher ground clearance,
  larger interiors and a commanding driving position, making them
  suitable for both city and off-road driving.
```

### Option 2 — Web app (Gradio)

```bash
python app.py
```

Then open the local URL printed in the terminal (usually
`http://127.0.0.1:7860`). In the app you can:

- Upload and preview a car image
- Click **Predict** (or it predicts automatically on upload)
- See the predicted body type + confidence
- See the Top-3 predictions with probabilities
- Read a short description of the predicted body type
- View a **Grad-CAM heatmap** showing which regions of the image the model
  focused on to make its decision

> ⚠️ You must run `python train.py` at least once before using `predict.py`
> or `app.py`, so that `saved_model/car_body_classifier.keras` exists.

---

## 📸 Screenshots

Add your own screenshots of the running app here:

```
screenshots/
  upload_screen.png
  prediction_result.png
  gradcam_example.png
```

_(Placeholders — replace with actual screenshots after running the app.)_

---

## ✨ Features Implemented

- ✔ Transfer Learning using MobileNetV2 (frozen base + custom head)
- ✔ Fine-tuning of top MobileNetV2 layers for extra accuracy
- ✔ Image preprocessing (resize, normalize, RGB conversion)
- ✔ Data augmentation (rotation, zoom, shift, flip, brightness)
- ✔ Model checkpoint saving (best validation accuracy)
- ✔ Early stopping to prevent overfitting
- ✔ Accuracy & loss graphs saved after training
- ✔ Automatic model saving to `saved_model/`
- ✔ Model loading during prediction (cached, loaded once)
- ✔ Confidence percentage for predictions
- ✔ Top-3 predictions with probabilities
- ✔ Responsive Gradio web interface
- ✔ Graceful handling of invalid/corrupted image uploads
- ✔ **Bonus:** Grad-CAM heatmap visualisation
- ✔ **Bonus:** Confusion matrix generation after evaluation
- ✔ **Bonus:** Classification report (precision/recall/F1)

---

## 🚀 Future Improvements

- Add more car body types (Coupe, Convertible, Pickup Truck, Van, etc.)
- Collect a larger, more diverse dataset for better generalisation
- Try other backbones (EfficientNet, ResNet50) and compare performance
- Deploy the app to Hugging Face Spaces / Render / AWS for public access
- Add a REST API (FastAPI/Flask) wrapper around `predict.py` for integration
  with other applications
- Add batch prediction support (classify multiple images at once)
- Add model explainability comparisons (Grad-CAM vs SHAP vs LIME)
- Convert the model to TensorFlow Lite for mobile deployment

---

## 🧰 Tech Stack

| Component        | Technology              |
|-------------------|--------------------------|
| Language           | Python 3.11+             |
| DL Framework       | TensorFlow / Keras       |
| Pretrained Model   | MobileNetV2 (ImageNet)   |
| Frontend           | Gradio                   |
| Visualization      | Matplotlib, Seaborn      |
| Evaluation         | scikit-learn             |

---

## 📄 License

This project is provided as-is for educational purposes (college projects,
learning, and portfolio building). Feel free to modify and extend it.
