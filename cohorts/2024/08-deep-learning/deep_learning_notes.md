# Introduction to Deep Learning

**Deep Learning** (DL) is a ML technique that uses ***deep neural networks*** (DNN's) as models.

DL can be used for both regression and classification problems that deal with non-tabular data such as images, video, audio,etc. 

# TensorFlow and Keras

## Intro

**TensorFlow** is a library for ML and AI. **Keras** is a library that provides a Python interface for TensorFlow, making it simpler to use. Keras used to be an independent library but has been absorved into TensorFlow.

```python
import tensorflow as tf
from tensorflow import keras
```

## Working with Images

```python
from tensorflow.keras.preprocessing.image import load_img

# filepath is the path to the file containing an image
img = load_img(filepath, target_size=(299, 299))
```

When loading an image with load_img(), the resulting object is a PIL image. PIL stands for Python Image Library; PIL used to be a library but it was abandoned and nowadays the Pillow library is used instead, but the image format is the same. PIL images can easily be converted to NumPy arrays of dtype=uint8 (unsigned integers of size 8 bits):

* `x = np.array(img)`

# Pre-trained convolutional neural networks

Instead of training a DNN from scratch, we can use a pre-trained network in order to speed up work. There are many pre-trained networks available online.

The standard training dataset for general image classification is **ImageNet**, a dataset with over a million images in 1000 different categories.

For this example we will use a Xception network pre-trained on ImageNet.

Along with image size, the model also expects the batch_size which is the size of the batches of data (default 32). If one image is passed to the model, then the expected shape of the model should be (1, 229, 229, 3).

```python
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

model = Xception(weights='imagenet', input_shape(299, 299, 3))

x = (1, 229, 229, 3)
X = np.array([x])

X = preprocess_input(X)

pred = model.predict(X)
decode_predictions(pred)
```

* `model.predict()` returns an array with 1000 values for each input image. Each value is the probability that the input image belongs to one of the 1000 categories in ImageNet.
* `decode_predictions()` is a convenient translation function that converts the prediction probabilities to human-readable output, in order to understand the categories.

# Convolutional Neural Networks

A Convolutional Neural Network (CNN) is a type of DNN that is well suited for dealing with images. The Xception network used in the previous section is an example of CNN.

A Convolution neural network has multiple hidden layers that help in extracting information from an image. The four important layers in CNN are:

1. Convolution layer (filters layer, outputs feature maps. Can be used for other data)
2. ReLU layer
3. Pooling layer
4. Fully connected layer (also called Dense layer. Specific to the dataset, cant be used for other data)

# Transfer Learning

Following are the steps to create train/validation data for model:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build image generator for training (takes preprocessing input function)
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load in train dataset into train generator
train_ds = train_gen.flow_from_directory(directory=path/to/train_imgs_dir, # Train images directory
                                         target_size=(150,150), # resize images to train faster
                                         batch_size=32, # 32 images per batch
                                         class_mode='categorical')

# Create image generator for validation
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load in image for validation
val_ds = val_gen.flow_from_directory(directory=path/to/val_imgs_dir, # Validation image directory
                                     target_size=(150,150),
                                     batch_size=32,
                                     shuffle=False) # False for validation
```
* `class_mode` specifies the kind of label arrays that are returned:
    * `categorical` is the default mode, for multicategory classification problems. Labels will be 2D one-hot encoded labels.
    * `binary` is for binary classification models. Labels will be 1D binary models.
    * There are more modes for other specific kind of problems (sparse, input) but they won't be covered here.
aa
Following are the steps to build model from a pretrained model:

```python
# Build base model
base_model = Xception(weights='imagenet',
                      include_top=False, # to create custom dense layer
                      input_shape=(150,150,3))

# Freeze the convolutional base by preventing the weights being updated during training
base_model.trainable = False

# Define expected image shape as input
inputs = keras.Input(shape=(150,150,3))

# Feed inputs to the base model
base = base_model(inputs, training=False) # set False because the model contains BatchNormalization layer

# Convert matrices into vectors using pooling layer
vectors = keras.layers.GlobalAveragePooling2D()(base)

# Create dense layer of 10 classes
outputs = keras.layers.Dense(10)(vectors)

# Create model for training
model = keras.Model(inputs, outputs)
```

Following are the steps to instantiate optimizer and loss function:

```python
# Define learning rate
learning_rate = 0.01

# Create optimizer
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Define loss function
loss = keras.losses.CategoricalCrossentropy(from_logits=True) # to keep the raw output of dense layer without applying softmax

# Compile the model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy']) # evaluation metric accuracy
```

The model is ready to train once it is defined and compiled:

```python
# Train the model, validate it with validation data, and save the training history
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
```

**Classes, function, and attributes**:
- `from tensorflow.keras.preprocessing.image import ImageDataGenerator`: to read the image data and make it useful for training/validation
- `flow_from_directory()`: method to read the images directly from the directory
- `next(train_ds)`: to unpack features and target variables
- `train_ds.class_indices`: attribute to get classes according to the directory structure
- `GlobalAveragePooling2D()`: accepts 4D tensor as input and operates the mean on the height and width dimensionalities for all the channels and returns vector representation of all images
- `CategoricalCrossentropy()`: method to produces a one-hot array containing the probable match for each category in multi classification
- `model.fit()`: method to train model
- `epochs`: number of iterations over all of the training data
- `history.history`: history attribute is a dictionary recording loss and metrics values (accuracy in our case) for at each epoch