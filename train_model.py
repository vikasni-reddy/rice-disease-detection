import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
import json
from sklearn.utils import class_weight

# ---------------- CONFIG ----------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

# ---------------- MODEL ----------------
def create_model(num_classes):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

# ---------------- TRAINING ----------------
def main():
    train_dir = "data/train"
    val_dir = "data/test"

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError("data/train and data/test folders are required")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # class weights
    weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(weights))

    model = create_model(train_gen.num_classes)
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.2),
        ModelCheckpoint(
            "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=12,
        class_weight=class_weights
    )

    # Save FINAL model
    model = tf.keras.models.load_model("best_model.keras")
    model.save("rice_disease_model.keras")

    # Save class indices
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    print("✅ Training complete")
    print("Saved: rice_disease_model.keras")

if __name__ == "__main__":
    main()
