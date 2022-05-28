# Dogs vs Cats - From Deep Learning with Python 2nd Edition

The repo is an example of the how to use Tensorflow 2.9.1 and some of the new packaging from the Tensorflow/Keras library.

## Versions (see requirements.txt)

tensorflow==2.9.1
keras==2.9.0
keras-preprocessing==1.1.2

## Setup

```shell
pip install pip-tools
pip-compile
pip install -r requirements.txt

cd <data directory>
kaggle competitions download -c dogs-vs-cats
```

Unzip the training folder.  This folder has both cat and dog jpg images with filenames of the form:

[cat | dog].<indexnumber>.jpg

## Create Dataset

Run the `create_dataset.py` script to create a smaller dataset for train/test/validation.

## Create Models

Run one or all of:

### Fast Feature Extraction

`fast_feature_extraction.py`

This model uses a pretrained ( imagenet weights) models, and removes the fully connected head.  
For each image, we call the predict method to obtain the last layer of 5,5,512 matrix.

We then train a new dense head to perform the actual classification.

Note that this method does some significant preprocessing of the images via the `predict` call.

### Basic 'from scratch' Model

`train_model.py`

This model is a basic, hand built, and fully trained model.

The script allows for the model to be built with/without image augmentation and with/without a final dropout layer.

The line allows for image augmentation and dropout.

    `model = build_model(with_augmentation=False, with_dropout=False)`

### Frozen Feature Exactor

`feature_extraction_freeze_conv.py`

This model will remove the head of the VGG16 model, and freeze the remaining convolutional layers.

A new dense head model will be created, and the entire new model will be trained from raw the images.

### Fast Feature extraction with Malaria Dataset

`fast_feature_extraction_cell_images.py`

##### Dataset

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

## Model Results

### Basic 'from scratch' Model

```text
Epoch 30/30
63/63 [==============================] - 38s 595ms/step - loss: 0.0341 - accuracy: 0.9915 - val_loss: 1.8174 - val_accuracy: 0.7550
```

```text
From Scratch Test accuracy: 0.684
```

### Basic Model with Data Augmentation and Dropout

```text
Epoch 30/30
63/63 [==============================] - 38s 607ms/step - loss: 0.3732 - accuracy: 0.8390 - val_loss: 0.4192 - val_accuracy: 0.8150
```

```text
From Scratch w/ Image Aug Test accuracy: 0.800
```


### Fast Feature Extraction without Image Augmentation

```text
Epoch 30/30
63/63 [==============================] - 2s 29ms/step - loss: 8.4480e-10 - accuracy: 1.0000 - val_loss: 5.1415 - val_accuracy: 0.9770
```

```text
From Fast Feature Extract Test accuracy: 0.975
```

## Cell Images Dataset

```text
Epoch 30/30
624/624 [==============================] - 16s 26ms/step - loss: 0.7086 - accuracy: 0.9828 - val_loss: 2.7155 - val_accuracy: 0.9637

```

```text
From Feature Extractor Cell Images Model Test accuracy: 0.965

```