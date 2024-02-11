import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model

st.header('Flower Classification')

flowers_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('./model_11-2024.h5')

def predict_img(img_path):
    input_img = tf.keras.utils.load_img(img_path, target_size=(180, 180))
    input_img_array = tf.keras.utils.img_to_array(input_img)
    input_img_exp_dim = tf.expand_dims(input_img_array, 0)

    pred = model.predict(input_img_exp_dim)
    result = tf.nn.softmax(pred[0])
    output = 'The flower belong to ' + flowers_names[np.argmax(result)]
    return output

upload_file = st.file_uploader('upload a flower image')
if upload_file is not None:
    with open(os.path.join('upload_dir', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    
    st.image(upload_file, width=200)

st.markdown(predict_img(upload_file))