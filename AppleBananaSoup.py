import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load images and labels into arrays (same as your original code)

# Path to the dataset directory
dataset_dir = '/Users/aditya/dataset'

# Load images and labels into arrays
def load_data(dataset_dir):
    images = []
    labels = []
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print("Class names: ", class_names)
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        print("Loading images from", class_dir)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            print("Loading image", img_path)
            if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    img = load_img(img_path, target_size=(150, 150))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(class_indices[class_name])
                    print("Loaded image", img_path)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels), class_indices

images, labels, class_indices = load_data(dataset_dir)
print("Images shape:", images.shape)

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_indices))

# Split the dataset into test set (5%) and remaining set (95%)
X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.05, random_state=42, stratify=labels)
print("Test set shape:", X_test.shape)

# Split the remaining set into training (70% of original data) and validation set (25% of original data)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25 / 0.95, random_state=42, stratify=y_temp)
print("Training set shape:", X_train.shape)

# Data augmentation and rescaling for the training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,  # Adjusted augmentation settings
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Adding brightness adjustment
    fill_mode='nearest'
)

# Only rescaling for the validation and test data
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches
train_generator = train_datagen.flow(X_train, y_train, batch_size=64)

# Flow validation images in batches
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=64)

# Flow test images in batches
test_generator = test_datagen.flow(X_test, y_test, batch_size=64)

# Model building
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Added batch normalization
model.add(layers.MaxPooling2D((2, 2)))

# Fully connected layers
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))  # Adjusted regularization
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))  # Adjusted regularization
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))  # Adjusted regularization
model.add(layers.Dense(len(class_indices), activation='softmax', kernel_regularizer=regularizers.l2(1e-4)))  # Adjusted regularization

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Adjusted learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.summary()

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 64,  # Number of batches per epoch
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(X_val) // 64,  # Number of batches for validation
    callbacks=[early_stopping]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], linestyle='--', label='Validation')  # Dotted line for validation
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], linestyle='--', label='Validation')  # Dotted line for validation
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=len(X_test) // 64)
print('Test accuracy:', test_acc)

predictions = model.predict(test_generator)

# Plotting a subplot
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    predicted_label = np.argmax(predictions[i])
    actual_label = np.argmax(y_test[i])
    plt.title(
        f'Predicted: {list(class_indices.keys())[predicted_label]}\nActual: {list(class_indices.keys())[actual_label]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

model.save('my_model.h5')

# Save the model in TensorFlow SavedModel format
model.save('my_model')

# Save the model in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
