import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import os
import cv2


model = models.load_model('.\Face_Emotion_Recognition\model_01.h5')

st.header('Human Emotion Recongition')
img_path = st.text_input('uploda an image')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

img = cv2.imread(img_path)[:,:,0]
img = cv2.resize(img, (48, 48))
img = np.invert(np.array([img]))

output = np.argmax(model.predict(img))
outcome = emotions[output]
stn = 'The emotion in the image is ' + str(outcome)
st.markdown(stn)

img_name = os.path.basename(img_path)
st.image('Image' + img_name, width=300)