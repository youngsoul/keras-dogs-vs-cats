from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.utils.image_dataset import image_dataset_from_directory
from pathlib import Path
import pandas as pd
import numpy as np


def build_data_augmentation():
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ])
    return data_aug


def create_datasets():
    target_dir = Path("./data/cats_vs_dogs_small")

    train_dataset = image_dataset_from_directory(
        target_dir / "train",
        image_size=(180, 180),
        batch_size=32
    )

    validation_dataset = image_dataset_from_directory(
        target_dir / "validation",
        image_size=(180, 180),
        batch_size=32
    )

    test_dataset = image_dataset_from_directory(
        target_dir / "test",
        image_size=(180, 180),
        batch_size=32
    )

    return train_dataset, validation_dataset, test_dataset


def create_feature_extractor():
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False
    )
    conv_base.trainable = False

    return conv_base


def create_classifier(conv_base):
    inputs = keras.Input(shape=(180, 180, 3))
    x = build_data_augmentation()(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    return model


def main():
    conv_base = create_feature_extractor()

    classifier_model = create_classifier(conv_base)

    classifier_model.compile(loss="binary_crossentropy",
                             optimizer="rmsprop",
                             metrics=['accuracy'])

    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"./models/feature_extraction_frozen.keras",
        save_best_only=True,
        monitor="val_loss"
    )

    train_ds, val_ds, test_ds = create_datasets()

    history = classifier_model.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        callbacks=[checkpoint_callback]
    )

    losses = pd.DataFrame(history.history)
    print(losses.head())
    losses.to_csv(f"./history/feature_extraction_frozen.csv")


if __name__ == '__main__':
    main()
