import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Define parameters
IMG_SIZE = 256
TARGET_SIZE = 224  # Resize images for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 2  # Binary classification
LEARNING_RATE = 0.001  # Lowered for better stability

# Function to load images from directory
def loadImagesFromDirectory(dataset, label):
    images, labels = [], []
    for imgFile in os.listdir(dataset):
        if imgFile.endswith(('.jpg', '.png')):  # Handle both JPG and PNG
            imgPath = os.path.join(dataset, imgFile)
            img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(IMG_SIZE, IMG_SIZE))
            imgArray = tf.keras.preprocessing.image.img_to_array(img)
            images.append(imgArray)
            labels.append(label)
    return np.array(images), np.array(labels)

# Function to load dataset
def load_dataset(dataset_path):
    pos_images, pos_labels = loadImagesFromDirectory(os.path.join(dataset_path, 'Flower'), 1)
    neg_images, neg_labels = loadImagesFromDirectory(os.path.join(dataset_path, 'NonFlower'), 0)
    
    # Merge and normalize images
    images = np.concatenate((pos_images, neg_images), axis=0) / 255.0
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    return images, labels

# Load dataset
dataset_path = "dataset"  # Update with your dataset path
images, labels = load_dataset(dataset_path)

# Split dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Resize images to match MobileNetV2 input size (224, 224)
X_train_resized = tf.image.resize(X_train, (TARGET_SIZE, TARGET_SIZE))
X_val_resized = tf.image.resize(X_val, (TARGET_SIZE, TARGET_SIZE))
X_test_resized = tf.image.resize(X_test, (TARGET_SIZE, TARGET_SIZE))

# Data Augmentation (no rescaling since images are already normalized)
datagen = ImageDataGenerator(
    rotation_range=40,  
    width_shift_range=0.3,  
    height_shift_range=0.3,  
    shear_range=0.3,  
    zoom_range=0.3,  
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train_resized.numpy())  # Convert to NumPy before fitting

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Freeze the base model layers
base_model.trainable = False

# Build the model by adding custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)  # Reduce overfitting
x = BatchNormalization()(x)  # Improve generalization
x = Dense(NUM_CLASSES, activation='softmax')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train_resized.numpy(), y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val_resized.numpy(), y_val),
    epochs=EPOCHS
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test_resized, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save("flower_model.h5")

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("flower_model.tflite", "wb") as f:
    f.write(tflite_model)

# Confusion Matrix & Classification Report
y_pred = model.predict(X_test_resized.numpy())
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
report = classification_report(y_test, y_pred_classes, target_names=["NonFlower", "Flower"])
print("\nClassification Report:\n", report)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xticks([0, 1], ['NonFlower', 'Flower'])
plt.yticks([0, 1], ['NonFlower', 'Flower'])
plt.xlabel('Predicted')
plt.ylabel('True')

# Annotate the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.title('Confusion Matrix')
plt.show()
