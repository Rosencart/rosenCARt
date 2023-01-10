import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create the model
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(43, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the data generators
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Define the directories for the training and test sets
train_set = train_datagen.flow_from_directory('train',
                                              target_size = (32, 32),
                                              batch_size = 32,
                                              class_mode = 'sparse')

test_set = test_datagen.flow_from_directory('test',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             class_mode = 'sparse')

# Train the model
history = model.fit(train_set,
                    epochs = 25,
                    validation_data = test_set)

# Save the model
model.save("model.h5")