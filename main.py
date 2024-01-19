


import glob
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from classification_network import classificator
from densenet_model import densenet_model
import tensorflow as tf
from feature_extracting import feature_encoding
from normalization import daugman_normalization
import keras.utils as image
from tensorflow.keras.utils import to_categorical
import time
from keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt

from segmentation import segment
from split_train_test_validation import prepare_data_to_fit
features_file = 'features.npy'
labels_file = 'labels.npy'
imgs_file = 'imgs.npy'
print("Available devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU USAGE")

img_size=(64,64)
def load_features_and_labels(): 
    dataset_path = 'CASIA-Iris-Thousand'
    features = []
    labels = []
    model = densenet_model()
    for filefilepath in glob.iglob('datasets/normalized_casia_224x224_segmented_clahe/*'):
        if filefilepath[-1] == 'g':
            print(filefilepath)
            split = filefilepath.split(".")
            label=int(split[0].split("\\")[1])
            img = cv2.imread(filefilepath)
            segmented_iris = image.img_to_array(img)
            segmented_iris = np.expand_dims(segmented_iris, axis=0)  # Add batch dimension
            segmented_iris = preprocess_input(segmented_iris)
            feature = model.predict(segmented_iris)
            features.append(feature)
         
            print(label)
            labels.append(label)
    return (np.array(features), np.array(labels))

if os.path.exists(features_file) and os.path.exists(labels_file):
    features = np.load(features_file)
    labels = np.load(labels_file)
else: 
    (features, labels) = load_features_and_labels()
    print("Saving features")
    np.save(features_file, features)
    print(labels)
    labels = np.array(labels)
    print("Saving labels")
    np.save(labels_file, labels)
#features = np.expand_dims(features, axis=-1)
print(labels)
features = features.reshape((-1, 7, 7, 1920))

num_classes = len(np.unique(labels))
print("Number of classes")
print(num_classes)
print("Number of features")
print(len(features))
labels = to_categorical(labels, num_classes=num_classes)
print("labels to_categorical")
print(labels)
print("feature_shape")
print(features.shape)
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data_to_fit(features, labels)
input_shape = features.shape
print("input_shape")
print(input_shape)

classifier = classificator(input_shape=input_shape, num_classes=num_classes)

start_time = time.time()
# Train the classifier
history = classifier.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=1000, verbose=1)

# Stop measuring time
end_time = time.time()

# Calculate the training time in seconds
training_time_seconds = end_time - start_time
# Convert training time to hours, minutes, and seconds
training_hours, remainder = divmod(training_time_seconds, 3600)
training_minutes, training_seconds = divmod(remainder, 60)

# Print the training time in a human-readable format
print(f"Training time: {int(training_hours):02d} hours, {int(training_minutes):02d} minutes, {int(training_seconds):02d} seconds")


loss, accuracy = classifier.evaluate(X_test, y_test, verbose=2)
print("Saving model")
classifier.save('iris_recognition_model.h5')
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss*100:.2f}%")

print(history.history.keys())
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# create error sublpot
plt.plot(history.history["loss"], label="train error")
plt.plot(history.history["val_loss"], label="validation error")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()