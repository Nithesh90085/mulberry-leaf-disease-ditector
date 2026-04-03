"""
One-click training script for Mulberry Leaf Disease Detector.
Run: python train.py

This script:
1. Downloads the PlantVillage dataset from Kaggle
2. Filters mulberry-relevant disease classes
3. Trains MobileNetV2 model
4. Saves mulberry_model.h5
"""

import os
import sys
import shutil
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications import MobileNetV2

# ── Config ──────────────────────────────────────────────────────────────────
# Allow passing dataset path as argument: python train.py --dataset C:\dataset
DATASET_DIR = "dataset"
for i, arg in enumerate(sys.argv):
    if arg == "--dataset" and i + 1 < len(sys.argv):
        DATASET_DIR = sys.argv[i + 1]
        break
MODEL_PATH  = "mulberry_model.h5"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20

# Map PlantVillage folder names → our display labels
# If you have a real mulberry dataset, put it in dataset/ with these folder names
CLASS_MAP = {
    "Healthy":          ["healthy", "Healthy"],
    "Leaf Rust":        ["rust", "Leaf_Rust", "leaf_rust"],
    "Leaf Spot":        ["spot", "Leaf_Spot", "leaf_spot", "cercospora"],
    "Powdery Mildew":   ["powdery", "Powdery_Mildew", "powdery_mildew"],
    "Bacterial Blight": ["blight", "Bacterial_Blight", "bacterial_blight"],
}

CLASSES = list(CLASS_MAP.keys())


def check_kaggle():
    """Check if kaggle CLI is available."""
    return shutil.which("kaggle") is not None


def download_dataset():
    """Download PlantVillage dataset via Kaggle API."""
    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        print(f"Dataset already exists at '{DATASET_DIR}/', skipping download.")
        return True

    if not check_kaggle():
        print("\n" + "="*60)
        print("MANUAL DATASET SETUP REQUIRED")
        print("="*60)
        print("\nOption A — Kaggle CLI (recommended):")
        print("  1. pip install kaggle")
        print("  2. Get API key from https://www.kaggle.com/settings")
        print("  3. Place kaggle.json in C:\\Users\\<you>\\.kaggle\\")
        print("  4. Re-run this script")
        print("\nOption B — Manual download:")
        print("  1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("  2. Download and extract")
        print(f"  3. Place disease folders inside: {os.path.abspath(DATASET_DIR)}/")
        print("     Example structure:")
        for cls in CLASSES:
            print(f"       {DATASET_DIR}/{cls}/  (put leaf images here)")
        print("\nOption C — Use your own images:")
        print(f"  Create folders under {DATASET_DIR}/ named exactly:")
        for cls in CLASSES:
            print(f"    {cls}")
        print("  Add at least 50 images per folder, then re-run.")
        print("="*60)
        return False

    print("Downloading dataset via Kaggle API...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.system("kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p dataset --unzip")
    return True


def prepare_dataset():
    """
    Find the actual image folders inside the downloaded dataset structure
    and map them to our class names.
    """
    if not os.path.exists(DATASET_DIR):
        return False

    print(f"\nUsing dataset path: {os.path.abspath(DATASET_DIR)}")

    # Known deep path from the PlantVillage Kaggle download
    deep_train = os.path.join(
        DATASET_DIR,
        "New Plant Diseases Dataset(Augmented)",
        "New Plant Diseases Dataset(Augmented)",
        "train"
    )

    # Walk to find any folder named 'train' with many subfolders
    actual_root = None
    if os.path.exists(deep_train):
        actual_root = deep_train
    else:
        for root, dirs, files in os.walk(DATASET_DIR):
            if os.path.basename(root).lower() == "train" and len(dirs) >= 5:
                actual_root = root
                break
        if actual_root is None:
            for root, dirs, files in os.walk(DATASET_DIR):
                if len(dirs) >= 5:
                    actual_root = root
                    break

    if actual_root is None:
        print("Could not find class folders inside dataset/")
        return False

    print(f"Found class folders at: {actual_root}")
    existing = os.listdir(actual_root)
    print(f"Sample folders: {existing[:8]}{'...' if len(existing)>8 else ''}")

    # Map PlantVillage folder names to our classes
    # PlantVillage uses format: Plant___Disease
    pv_map = {
        "Healthy":          ["healthy"],
        "Leaf Rust":        ["rust", "leaf_rust"],
        "Leaf Spot":        ["spot", "cercospora", "leaf_spot"],
        "Powdery Mildew":   ["powdery_mildew", "powdery"],
        "Bacterial Blight": ["bacterial_blight", "blight"],
    }

    for target_class, aliases in pv_map.items():
        dst = os.path.join(DATASET_DIR, target_class)
        if os.path.exists(dst):
            print(f"  '{target_class}' already exists, skipping.")
            continue
        for folder in existing:
            folder_lower = folder.lower()
            if any(alias in folder_lower for alias in aliases):
                src = os.path.join(actual_root, folder)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                    print(f"  Mapped '{folder}' → '{target_class}'")
                    break

    matched = [c for c in CLASSES if os.path.exists(os.path.join(DATASET_DIR, c))]

    # If still no matches, just train on all raw PlantVillage classes
    if len(matched) < 2:
        print("\nNo mulberry-specific classes mapped. Training on full PlantVillage dataset.")
        return actual_root

    print(f"Ready classes ({len(matched)}): {matched}")
    return DATASET_DIR


def build_model(num_classes):
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train():
    print("\n" + "="*60)
    print("MULBERRY LEAF DISEASE MODEL TRAINING")
    print("="*60)

    # Step 1: Dataset
    if not download_dataset():
        return

    dataset_path = prepare_dataset()
    if not dataset_path:
        print("\nNot enough class folders found. Please add images manually.")
        return

    # Step 2: Load data
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    print("\nLoading training data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes found ({num_classes}): {class_names}")

    # Save class names so app.py can use them
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print("Saved class_names.txt")

    train_ds = train_ds.map(
        lambda x, y: (augment(x / 255.0, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (x / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Step 3: Train
    print(f"\nTraining for up to {EPOCHS} epochs (early stopping enabled)...")
    model = build_model(num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Step 4: Fine-tune top layers of base model
    print("\nFine-tuning top 30 layers of MobileNetV2...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print("\nRestart app.py to use the trained model.")


if __name__ == "__main__":
    train()
