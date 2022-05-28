from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.utils.image_dataset import image_dataset_from_directory
from pathlib import Path
import pandas as pd

def build_data_augmentation():
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ])
    return data_aug

def build_model(with_augmentation = False, with_dropout=False):
    data_agumentation = build_data_augmentation() # step 2

    inputs = keras.Input(shape=(180, 180, 3))
    if with_augmentation:
        x = data_agumentation(inputs)
        x = layers.Rescaling(1. / 255)(x)
    else:
        x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)

    x = layers.Flatten()(x)
    if with_dropout:
        x = layers.Dropout(0.5)(x) # step 2

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


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

def main():
    filename = "convnet_from_scratch"
    model = build_model(with_augmentation=False, with_dropout=False)

    print(model.summary())

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=['accuracy'])

    train_ds, val_ds, test_ds = create_datasets()

    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"./models/{filename}.keras",
        save_best_only=True,
        monitor="val_loss"
    )
    earlystopping_callback = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=False
    )
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=[checkpoint_callback]
    )

    losses = pd.DataFrame(history.history)
    print(losses.head())
    losses.to_csv(f"./history/{filename}.csv")

if __name__ == '__main__':
    main()
