from tensorflow.keras.models import load_model

model = load_model('cat_dog_classifier.h5')

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
