from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.utils.image_dataset import image_dataset_from_directory
from pathlib import Path
import pandas as pd
import numpy as np

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


def get_features_and_labels(conv_base, dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        # images - batches of shape, 32,180,180,3
        # labels = array, size 32, of 1s and 0s
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        # features: 32, 5,5,512 dimension
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def create_feature_extractor():
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(180,180,3)
    )

    # the conv_base final layer returns a 5,5,512 dataset
    return conv_base

def create_classifier():
    inputs = keras.Input(shape=(5,5,512))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    return model


def main():
    conv_base = create_feature_extractor()

    train_ds, val_ds, test_ds = create_datasets()

    train_features, train_labels = get_features_and_labels(conv_base, train_ds)
    val_features, val_labels = get_features_and_labels(conv_base, val_ds)
    test_features, test_labels = get_features_and_labels(conv_base, test_ds)

    classifier_model = create_classifier()
    classifier_model.compile(loss="binary_crossentropy",
                             optimizer="rmsprop",
                             metrics=['accuracy'])

    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"./models/fast_feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss"
    )

    history = classifier_model.fit(
        train_features, train_labels,
        epochs=30,
        validation_data=(val_features, val_labels),
        callbacks=[checkpoint_callback]
    )

    losses = pd.DataFrame(history.history)
    print(losses.head())
    losses.to_csv(f"./history/fast_feature_extraction.csv")




if __name__ == '__main__':
    main()
