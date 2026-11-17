"""
class_names.py
----------------
Centralised place for class labels and their human-readable descriptions.

Keeping this in one file means train.py, predict.py and app.py always agree
on the class order and on the text shown to the user. If you ever add a new
body type (e.g. "coupe"), you only need to:
  1. Add a new folder under dataset/train, dataset/validation, dataset/test
  2. Add the class name + description below
"""

# NOTE: The order here MUST match the alphabetical folder order that
# TensorFlow's image_dataset_from_directory / ImageDataGenerator uses,
# which is simply sorted(os.listdir(train_dir)).
# For our four folders (hatchback, mpv, sedan, suv) the alphabetical
# order is exactly the order below.
CLASS_NAMES = ["hatchback", "mpv", "sedan", "suv"]

# Friendly display names (used only for showing results to the user)
DISPLAY_NAMES = {
    "hatchback": "Hatchback",
    "mpv": "MPV/MUV",
    "sedan": "Sedan",
    "suv": "SUV",
}

# Short description shown under the prediction in the Gradio app
CLASS_DESCRIPTIONS = {
    "hatchback": (
        "Hatchbacks are compact cars with a rear door that swings upward "
        "to provide access to a shared passenger/cargo area. They are "
        "fuel-efficient, easy to park and ideal for city driving."
    ),
    "mpv": (
        "MPVs (Multi-Purpose Vehicles) / MUVs (Multi-Utility Vehicles) are "
        "designed to maximise interior space and seating capacity, usually "
        "offering 6-8 seats. They are popular for large families and group travel."
    ),
    "sedan": (
        "Sedans are three-box vehicles with distinct compartments for the "
        "engine, passengers and boot (trunk). They offer a comfortable ride, "
        "good boot space and a balanced driving experience."
    ),
    "suv": (
        "SUVs (Sport Utility Vehicles) usually have higher ground clearance, "
        "larger interiors and a commanding driving position, making them "
        "suitable for both city and off-road driving."
    ),
}


def get_display_name(class_name: str) -> str:
    """Return the friendly display name for a raw class name."""
    return DISPLAY_NAMES.get(class_name, class_name.title())


def get_description(class_name: str) -> str:
    """Return the short description for a raw class name."""
    return CLASS_DESCRIPTIONS.get(class_name, "No description available.")
