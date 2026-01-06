# contents of file
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

# Set image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

def create_model(num_classes=NUM_CLASSES, fine_tune_at=None):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    if fine_tune_at is not None:
        # Unfreeze from "fine_tune_at" layer index onwards
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    return model

def main():
    train_dir = 'data/train'
    val_dir = 'data/test'  # kept same as your project
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise ValueError("Expected data directories 'data/train' and 'data/test' to exist.")

    # Use EfficientNet preprocess_input during training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    # Compute class weights to balance training if classes are imbalanced
    classes = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", class_weights)

    model = create_model(num_classes=train_generator.num_classes, fine_tune_at=None)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    ]

    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    history = model.fit(
        train_generator,
        epochs=12,  # initial training: train top head only
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    # Fine-tune: unfreeze last layers and continue training if needed
    # Unfreeze from a late stage and continue training with a lower LR
    model = tf.keras.models.load_model('best_model.h5')  # load best head-only model
    base_model = model.layers[1] if len(model.layers) > 1 else None
    # Fine-tune last N layers of EfficientNetB0
    if base_model is not None:
        # Make entire base trainable then freeze earlier layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks_ft = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
            ModelCheckpoint('best_model_finetuned.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]

        history_ft = model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            callbacks=callbacks_ft,
            class_weight=class_weights,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1
        )

    # Save final model (finetuned if available)
    final_model_path = 'rice_disease_model.h5'
    if os.path.exists('best_model_finetuned.h5'):
        tf.keras.models.load_model('best_model_finetuned.h5').save(final_model_path)
    else:
        tf.keras.models.load_model('best_model.h5').save(final_model_path)

    # Save class indices
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    # Optionally save history
    print("Training complete. Final model saved to", final_model_path)

if __name__ == "__main__":
    # GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

    main()