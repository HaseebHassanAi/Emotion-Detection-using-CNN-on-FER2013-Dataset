"""
Facial Emotion Recognition using CNN (FER2013 Dataset)
======================================================
- Uses CNN with L1/L2 regularization, Dropout, and BatchNormalization
- Includes Data Augmentation
- Implements EarlyStopping and ReduceLROnPlateau
"""

# ===============================
# Imports
# ===============================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from warnings import filterwarnings

filterwarnings("ignore")


# ===============================
# Data Preparation
# ===============================
IMG_SIZE = (48, 48)       # FER2013 images are 48x48
BATCH_SIZE = 64
DATA_DIR = "/kaggle/input/fer2013"   # Change this if needed

# Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=[0.8, 1.2]
)

# Only rescale for test/validation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical"
)

NUM_CLASSES = train_generator.num_classes
print("Class mapping:", train_generator.class_indices)


# ===============================
# Model Definition
# ===============================
def build_model(input_shape, num_classes):
    l2_reg = regularizers.l2(0.001)

    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=input_shape, kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 4
        Conv2D(256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Dense Layers
        Flatten(),
        Dense(512, activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation="relu", kernel_regularizer=l2_reg),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model


# ===============================
# Training
# ===============================
input_shape = (48, 48, 1)
model = build_model(input_shape, NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=100,
    callbacks=[early_stopping, reduce_lr]
)


# ===============================
# Evaluation
# ===============================
loss, acc = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")


# ===============================
# Single Image Prediction
# ===============================
def predict_emotion(model, img_path, class_labels):
    """Predict emotion for a single image."""
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    return class_labels[pred_class], confidence


# Reverse mapping {idx: class_name}
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Example prediction
test_img = os.path.join(DATA_DIR, "train/fear/Training_10031494.jpg")
emotion, conf = predict_emotion(model, test_img, class_labels)
print(f"Predicted Emotion: {emotion} ({conf:.2f} confidence)")