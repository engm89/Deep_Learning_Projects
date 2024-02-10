import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import  load_model

labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
IMG_WIDTH = 180
IMG_HIGHT = 180

model = load_model('./model_01.h5')
#img = './Fruits_Vegetables/test/apple/Image_3.jpg'
img = st.text_input("Uploda an image", 'Image_3.jpg')
st.header('Image Classification')
img = tf.keras.utils.load_img(img, target_size=(IMG_HIGHT, IMG_WIDTH))
img_array = tf.keras.utils.array_to_img(img)
img_dims = tf.expand_dims(img_array, 0)
pred = model.predict(img_dims)
output = tf.nn.softmax(pred)


st.image(img, width=200)
st.write(f'The image is {labels[np.argmax((output))]}')
