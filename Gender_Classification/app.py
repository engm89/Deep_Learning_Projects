import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model

labels = ['female', 'male']
IMG_HIGHT = 64
IMG_WIDTH = 64

model = load_model('./model_1.h5')

st.header('Female/Male')

def predict_func(img_path):
    input_img = tf.keras.utils.load_img(img_path, target_size=(IMG_HIGHT, IMG_WIDTH))
    input_img_to_array = tf.keras.utils.img_to_array(input_img)
    input_img_exp_dims = tf.expand_dims(input_img_to_array, 0)

    pred = model.predict(input_img_exp_dims)
    result = tf.nn.softmax(pred[0])
    output = 'The uploaded image is ' + labels[np.argmax(result)]
    return output

uploaded_file = st.file_uploader('upload a image')
if uploaded_file is not None:
    with open(os.path.join('upload_dir', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)

st.markdown(predict_func(uploaded_file))
