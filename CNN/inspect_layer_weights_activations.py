from tensorflow.keras.models import load_model

model = load_model('cat_dog_classifier.h5')

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

first_layer_weights = model.layers[0].get_weights()[0]
print("First layer weights shape:", first_layer_weights.shape)

# Visualize filters in first conv layer
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(16):  # show first 16 filters
    plt.subplot(4, 4, i+1)
    plt.imshow(first_layer_weights[:, :, 0, i], cmap='viridis')
    plt.axis('off')
plt.show()
