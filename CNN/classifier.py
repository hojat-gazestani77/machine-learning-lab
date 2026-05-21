from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Only rescaling for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Point generators to your directories
train_generator = train_datagen.flow_from_directory(
    'Dogs-vs-Cats/training_set/training_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'Dogs-vs-Cats/test_set/test_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
