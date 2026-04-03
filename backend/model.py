"""
Mulberry Leaf Disease Detection - CNN Model
Based on research: MobileNetV2 transfer learning approach
Reference: https://www.frontiersin.org/articles/10.3389/fpls.2023.1175515/full
Dataset: Mulberry leaf images (Healthy, Leaf Rust, Leaf Spot, Powdery Mildew)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Disease classes — overridden by class_names.txt if model is trained
CLASSES = [
    "Healthy",
    "Leaf Rust",
    "Leaf Spot",
    "Powdery Mildew",
    "Bacterial Blight"
]

# Load class names from training output if available
_class_file = os.path.join(os.path.dirname(__file__), "class_names.txt")
if os.path.exists(_class_file):
    with open(_class_file) as f:
        CLASSES = [line.strip() for line in f if line.strip()]

IMG_SIZE = (224, 224)
MODEL_PATH = "mulberry_model.h5"

# ── Leaf validator using ImageNet MobileNetV2 ──────────────────────────────
# ImageNet synset IDs that correspond to plants / leaves / nature
_PLANT_SYNSETS = {
    "n11939491",  # daisy
    "n12057211",  # yellow lady's slipper
    "n13133613",  # ear (of corn)
    "n11978233",  # agaric (mushroom — borderline, keep)
    "n12144580",  # corn
    "n12267677",  # acorn
    "n12620546",  # hip (rose hip)
    "n12768682",  # buckeye / horse chestnut
    "n13652335",  # fig
    "n13054560",  # bolete mushroom
    "n07753592",  # banana
    "n07747607",  # orange
    "n07749582",  # lemon
    "n07753275",  # pineapple
    "n07714571",  # broccoli
    "n07714990",  # cauliflower
    "n07715103",  # head cabbage
    "n07716358",  # zucchini
    "n07716906",  # spaghetti squash
    "n07717410",  # acorn squash
    "n07717556",  # butternut squash
    "n07718472",  # cucumber
    "n07718747",  # artichoke
    "n07720875",  # bell pepper
    "n07730033",  # cardoon
    "n07734744",  # mushroom
    "n12985857",  # coral fungus
    "n12998815",  # agaric
    "n13040303",  # lichen
    "n13044778",  # hen-of-the-woods
    "n13052670",  # earthstar
    "n13054560",  # bolete
}

_imagenet_validator = None

def _get_validator():
    global _imagenet_validator
    if _imagenet_validator is None:
        _imagenet_validator = MobileNetV2(weights="imagenet", include_top=True)
    return _imagenet_validator

def is_leaf_image(image_path: str, top_n: int = 15) -> bool:
    """
    Two-stage check:
    1. ImageNet: reject if confidently a non-plant object
    2. Color check: leaves have significant green/yellow/brown tones
    """
    # ── Stage 1: ImageNet check ──────────────────────────────────────────
    validator = _get_validator()
    img = keras.utils.load_img(image_path, target_size=(224, 224))
    arr = keras.utils.img_to_array(img)
    arr_batch = np.expand_dims(arr, axis=0)
    arr_batch = preprocess_input(arr_batch)

    preds = validator.predict(arr_batch, verbose=0)
    top = decode_predictions(preds, top=top_n)[0]

    _PLANT_KEYWORDS = (
        "leaf", "plant", "flower", "tree", "shrub", "herb",
        "fern", "moss", "grass", "weed", "vine", "foliage",
        "blossom", "petal", "stem", "branch", "twig", "bark",
        "mulberry", "berry", "fruit", "vegetable", "crop",
        "rapeseed", "hay", "corn", "cabbage", "lettuce", "spinach",
        "algae", "seaweed", "reed", "bamboo", "palm", "maple",
        "oak", "pine", "willow", "fig", "tobacco", "cotton",
        "daisy", "acorn", "cucumber", "zucchini", "artichoke",
    )

    _REJECT_KEYWORDS = (
        "person", "people", "man", "woman", "child", "baby", "face",
        "car", "truck", "bus", "vehicle", "bicycle", "motorcycle", "cab",
        "dog", "cat", "bird", "fish", "bear", "lion", "tiger", "elephant",
        "phone", "computer", "laptop", "keyboard", "screen", "television",
        "chair", "table", "sofa", "bed", "building", "house", "street",
        "pizza", "burger", "sandwich", "cake", "bread", "hot_dog",
        "bottle", "cup", "bowl", "plate", "gun", "knife",
    )

    imagenet_is_plant = False
    imagenet_is_object = False

    for class_id, label, prob in top:
        label_lower = label.lower()
        if class_id in _PLANT_SYNSETS or any(kw in label_lower for kw in _PLANT_KEYWORDS):
            imagenet_is_plant = True
            break
        if any(kw in label_lower for kw in _REJECT_KEYWORDS):
            imagenet_is_object = True

    # Confidently identified as a non-plant object → reject immediately
    if imagenet_is_object and not imagenet_is_plant:
        return False

    # ── Stage 2: Color check ─────────────────────────────────────────────
    # Load raw pixels (0-255) separately — arr was preprocessed to -1..1
    raw_img = keras.utils.load_img(image_path, target_size=(224, 224))
    raw_arr = keras.utils.img_to_array(raw_img)  # shape (224,224,3), values 0-255
    pixels = raw_arr.reshape(-1, 3)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # Green pixels: G channel dominant
    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 60)
    # Yellow pixels: R and G both high, B low
    yellow_mask = (r > 120) & (g > 120) & (b < 100) & (np.abs(r.astype(int) - g.astype(int)) < 60)
    # Brown pixels: R dominant, moderate G, low B
    brown_mask = (r > 80) & (g > 40) & (b < 80) & (r > g) & (r > b)

    total = len(pixels)
    leaf_color_ratio = (green_mask.sum() + yellow_mask.sum() + brown_mask.sum()) / total

    # At least 20% of the image should be leaf-colored
    if leaf_color_ratio < 0.20:
        return False

    return True

# ──────────────────────────────────────────────────────────────────────────


def build_model(num_classes=5):
    """
    Transfer learning with MobileNetV2 backbone.
    Achieves ~95% accuracy as per CNN-ViT research (PLOS ONE, 2024).
    """
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(dataset_path, epochs=20, batch_size=32):
    """
    Train the model on your dataset.
    Dataset structure expected:
        dataset_path/
            Healthy/
            Leaf Rust/
            Leaf Spot/
            Powdery Mildew/
            Bacterial Blight/

    Recommended datasets:
    - Kaggle Mulberry Leaf Disease: https://www.kaggle.com/datasets/
    - PlantVillage Dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
    """
    # Data augmentation pipeline
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Normalize + augment
    train_ds = train_ds.map(
        lambda x, y: (augment(x / 255.0, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (x / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    num_classes = len(train_ds.class_names) if hasattr(train_ds, 'class_names') else 5
    model = build_model(num_classes=num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"Model saved to {MODEL_PATH}")
    return model, history


def load_model():
    """Load trained model or return a demo model if not trained yet."""
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    print("WARNING: No trained model found. Using untrained model for demo.")
    print("Run train_model() with your dataset to get accurate predictions.")
    return build_model()


def preprocess_image(image_path):
    """Preprocess image for model inference."""
    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


CONFIDENCE_THRESHOLD = 60.0  # Below this % → not a mulberry leaf


def predict(image_path, model=None):
    """
    Predict disease from leaf image.
    Returns: dict with disease name, confidence, and treatment info.
    Raises ValueError if the image doesn't look like a mulberry leaf.
    """
    if model is None:
        model = load_model()

    # Gate: reject non-leaf images before running the disease model
    if not is_leaf_image(image_path):
        raise ValueError(
            "Invalid input: This doesn't appear to be a mulberry leaf. "
            "Please upload a clear image of a mulberry leaf."
        )

    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx]) * 100

    disease = CLASSES[predicted_idx] if predicted_idx < len(CLASSES) else "Unknown"

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "all_predictions": {
            CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(min(len(CLASSES), len(predictions[0])))
        },
        "treatment": TREATMENTS.get(disease, "Consult an agricultural expert."),
        "severity": get_severity(confidence),
        "disease_info": DISEASE_INFO.get(disease, {})
    }


def get_severity(confidence):
    if confidence >= 85:
        return "High Confidence"
    elif confidence >= 65:
        return "Moderate Confidence"
    else:
        return "Low Confidence - Manual inspection recommended"


DISEASE_INFO = {
    "Healthy": {
        "cause": None,
        "why": None,
        "solution": None
    },
    "Leaf Rust": {
        "cause": "Fungal infection by Cerotelium fici (formerly Phakopsora mori)",
        "why": [
            "High humidity and warm temperatures (25–30°C)",
            "Poor air circulation between plants",
            "Overhead irrigation wetting the leaves",
            "Infected plant debris left on the ground",
            "Spores spread by wind and rain splashes"
        ],
        "solution": [
            "Spray mancozeb or copper oxychloride fungicide every 10–14 days",
            "Remove and burn all infected leaves immediately",
            "Avoid overhead watering — use drip irrigation",
            "Improve spacing between plants for better airflow",
            "Apply neem oil spray as an organic preventive measure",
            "Clear fallen leaf debris from the base of plants"
        ]
    },
    "Leaf Spot": {
        "cause": "Fungal infection by Cercospora moricola or Phyllosticta mori",
        "why": [
            "Prolonged leaf wetness from rain or irrigation",
            "High humidity above 80% for extended periods",
            "Overcrowded planting reducing air circulation",
            "Infected seeds or planting material",
            "Fungal spores surviving in soil and plant debris"
        ],
        "solution": [
            "Apply copper-based fungicide (Bordeaux mixture) every 2 weeks",
            "Remove infected leaves and destroy them — do not compost",
            "Ensure proper drainage to avoid waterlogging",
            "Avoid wetting foliage during irrigation",
            "Maintain plant spacing for adequate air circulation",
            "Apply preventive fungicide spray at the start of humid season"
        ]
    }
}

TREATMENTS = {
    "Healthy": "Leaf is disease-free. Continue regular care and monitoring.",
    "Leaf Rust": "Apply mancozeb or copper-based fungicide. Remove infected leaves. Avoid overhead irrigation.",
    "Leaf Spot": "Apply copper-based fungicide (Bordeaux mixture). Remove infected leaves. Ensure proper drainage.",
    "Diseased": "Leaf shows signs of disease. Apply appropriate fungicide and consult an agricultural expert.",
    "Powdery Mildew": "Apply sulfur-based fungicides or potassium bicarbonate. Improve air circulation.",
    "Bacterial Blight": "Apply copper-based bactericides. Remove infected plant parts immediately."
}


def fine_tune_on_sample(model, image_path, correct_label, classes):
    """
    Fine-tune model on a single corrected sample using gradient descent.
    Uses a very small learning rate to avoid catastrophic forgetting.
    Saves updated model to disk.
    """
    if correct_label not in classes:
        return model

    label_idx = classes.index(correct_label)
    num_classes = len(classes)

    # One-hot encode the correct label
    label = np.zeros((1, num_classes), dtype=np.float32)
    label[0][label_idx] = 1.0

    img_array = preprocess_image(image_path)

    # Unfreeze top layers only for fine-tuning
    for layer in model.layers[-6:]:
        layer.trainable = True

    # Very low LR to nudge without forgetting
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train for 3 steps on this single sample with augmentation
    augmented = []
    import tensorflow as tf
    img_tensor = tf.constant(img_array)
    for _ in range(8):
        aug = tf.image.random_flip_left_right(img_tensor)
        aug = tf.image.random_flip_up_down(aug)
        aug = tf.image.random_brightness(aug, 0.15)
        aug = tf.image.random_contrast(aug, 0.85, 1.15)
        augmented.append(aug.numpy())

    x_batch = np.concatenate(augmented, axis=0)
    y_batch = np.tile(label, (8, 1))

    model.fit(x_batch, y_batch, epochs=3, verbose=0)

    # Save updated model
    model.save(MODEL_PATH)
    print(f"Model updated with feedback: {correct_label}")
    return model
