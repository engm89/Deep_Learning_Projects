import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

import warnings

warnings.filterwarnings('ignore')


# ! kaggle datasets -d kritikseth/fruit-and-vegetable-image-recognition
# from zipfile import ZipFile

# data_ds = './fruit-and-vegetable-image-recognition.zip'

# with ZipFile(data_ds, 'r') as zip:
#     zip.extractall()
#     print('The data-set is extracted')

IMG_WIDTH = 180
IMG_HIGHT = 180
EPOCHS_SIZE = 25

data_train_path = './Fruits_Vegetables/train/'
data_test_path = './Fruits_Vegetables/test/'
data_val_path = './Fruits_Vegetables/validation'

data_train = tf.keras.utils.image_dataset_from_directory(
    directory=data_train_path,
    shuffle=True,
    batch_size=32,
    image_size=(IMG_WIDTH, IMG_HIGHT),
    validation_split=False
)
data_cat = data_train.class_names
print(data_cat)

data_val = tf.keras.utils.image_dataset_from_directory(
    directory=data_val_path,
    shuffle=False,
    image_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=32,
    validation_split=False
)

data_test = tf.keras.utils.image_dataset_from_directory(
    directory=data_test_path,
    shuffle=False,
    image_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=32,
    validation_split=False
)

# plt.figure(figsize=(8, 8))
# for img, labels in data_train.take(1):
#     for i in range(9):
#         plt.subplot(3, 3, i+1)
#         plt.imshow(img[i].numpy().astype('uint8'))
#         plt.title(data_cat[labels[i]])
#         plt.axis('off')
# plt.show()

model = tf.keras.Sequential([
    layers.Rescaling(scale=1. / 255),

    layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),

    layers.Dense(128),
    layers.Dense(len(data_cat))
])

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

#history = model.fit(x=data_train, validation_data=data_val, epochs=EPOCHS_SIZE)

#model.save('model_01.h5')

saved_model = models.load_model('./model_01.h5')

img = './Fruits_Vegetables/test/apple/Image_3.jpg'
img = tf.keras.utils.load_img(img, target_size=(IMG_HIGHT, IMG_WIDTH))
img_array = tf.keras.utils.array_to_img(img)
img_dims = tf.expand_dims(img_array, 0)
pred = saved_model.predict(img_dims)

output = tf.nn.softmax(pred)
print(f'The image is {data_cat[np.argmax(output)]}, with accuracy of {np.max(output)}')






























