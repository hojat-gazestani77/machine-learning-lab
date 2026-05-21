import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cat_dog_classifier.h5')

img_path = 'catt.jpeg'  # Change path as needed
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("It's a dog!")
elif prediction[0][0] < 0.5:
    print("It's a cat!")
else:
    print("None of them")
