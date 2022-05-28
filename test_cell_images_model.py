from tensorflow import keras
from keras.utils.image_dataset import image_dataset_from_directory
from pathlib import Path
from keras.models import load_model
import numpy as np
import tensorflow as tf

image_height = 64
image_width = 64
base_model_name = "fast_feature_extraction_cell_images_64"

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

def create_datasets():
    target_dir = Path("/Volumes/MacBackup/ml_datasets/cell_images")

    test_dataset = image_dataset_from_directory(
        target_dir / "test",
        image_size=(image_width, image_height),
        batch_size=32
    )

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return test_ds

def create_feature_extractor():
    conv_base = keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_width,image_height,3)
    )

    return conv_base

def test_fast_feature_extract():
    test_ds = create_datasets()
    conv_base = create_feature_extractor()
    test_features, test_labels = get_features_and_labels(conv_base, test_ds)

    test_model = load_model(f'./models/{base_model_name}.keras')

    test_loss, test_acc = test_model.evaluate(test_features, test_labels)
    print(f"From Feature Extractor Cell Images Model Test accuracy: {test_acc:.3f}")



def main():
    test_fast_feature_extract()

if __name__ == '__main__':
    main()