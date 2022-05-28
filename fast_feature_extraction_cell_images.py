from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras import callbacks
from keras.utils.image_dataset import image_dataset_from_directory
from pathlib import Path
import pandas as pd
import numpy as np

image_height = 64
image_width = 64
epochs = 30
base_model_name = "fast_feature_extraction_cell_images_64"
target_dir = Path("/Volumes/MacBackup/ml_datasets/cell_images")


def create_datasets2():

    train_ds = image_dataset_from_directory(
        target_dir / "train",
        labels='inferred',
        label_mode='binary',
        validation_split=0.2,
        subset="training",
        shuffle=True,
        seed=123,
        image_size=(image_width, image_height),
        batch_size=32)

    val_ds = image_dataset_from_directory(
        target_dir / "train",
        labels='inferred',
        label_mode='binary',
        validation_split=0.2,
        subset="validation",
        shuffle=True,
        seed=123,
        image_size=(image_width, image_height),
        batch_size=32)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def create_datasets():

    train_dataset = image_dataset_from_directory(
        target_dir / "train",
        image_size=(image_width, image_height),
        batch_size=32
    )

    validation_dataset = image_dataset_from_directory(
        target_dir / "test",
        image_size=(image_width, image_height),
        batch_size=32
    )

    return train_dataset, validation_dataset


def get_features_and_labels(conv_base, dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        # images - batches of shape, 32,180,180,3
        # labels = array, size 32, of 1s and 0s
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        # features: 32, 5,5,512 dimension for a 180x180 image width and height
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def create_feature_extractor():
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_width,image_height,3)
    )

    return conv_base

def create_classifier(input_shape=None):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    return model


def main():
    conv_base = create_feature_extractor()

    print(conv_base.summary())
    output_shape_width = conv_base.layers[-1].output_shape[1]
    output_shape_height = conv_base.layers[-1].output_shape[2]
    output_shape_depth = conv_base.layers[-1].output_shape[3]

    train_ds, val_ds = create_datasets2()

    train_features, train_labels = get_features_and_labels(conv_base, train_ds)
    val_features, val_labels = get_features_and_labels(conv_base, val_ds)

    classifier_model = create_classifier(input_shape=(output_shape_width,output_shape_height, output_shape_depth))
    classifier_model.compile(loss="binary_crossentropy",
                             optimizer="rmsprop",
                             metrics=['accuracy'])

    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"./models/{base_model_name}.keras",
        save_best_only=True,
        monitor="val_loss"
    )

    history = classifier_model.fit(
        train_features, train_labels,
        epochs=epochs,
        validation_data=(val_features, val_labels),
        callbacks=[checkpoint_callback]
    )

    losses = pd.DataFrame(history.history)
    print(losses.head())
    losses.to_csv(f"./history/{base_model_name}.csv")


if __name__ == '__main__':
    main()
