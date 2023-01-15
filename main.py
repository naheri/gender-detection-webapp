# Making a Single Prediction
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array

def get_gender_from_path(image_path):
    # load model
    model = load_model('model_0.968.h5')
    img = load_img(image_path, target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return 'female' if model.predict(img) == 0 else 'male'