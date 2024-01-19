import os
import cv2
from keras.models import load_model
import numpy as np
from normalization import daugman_normalization
from keras.applications.densenet import preprocess_input

from segmentation import segment



# def pretrained_model():
#     model = VGG16(weights='imagenet', include_top=False)
#     model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

#     model.summary()

#     return model

#     # Загрузка модели, если файл существует
# model = load_model("iris_recognition_model.h5")
# im = cv2.imread("F:/repo/iris_rec/CASIA-Iris-Thousand/001/L/S5001L08.jpg", 0)

# #  segmentation
# segmented_iris, cirpupil, ciriris = segment(im)

# # segmented_iris = cv2.resize(segmented_iris, (224, 224))  # Resize to match Densenet input
# # segmented_iris = cv2.cvtColor(segmented_iris, cv2.COLOR_GRAY2RGB)
# # segmented_iris = img_to_array(segmented_iris)
# # segmented_iris = np.expand_dims(segmented_iris, axis=0)  # Add batch dimension
# # segmented_iris = preprocess_input(segmented_iris)



# # # normalization
# normalized_iris = daugman_normalization(segmented_iris, 224, 224, cirpupil[2], ciriris[2]-cirpupil[2])
# normalized_iris = cv2.cvtColor(normalized_iris, cv2.COLOR_BGR2GRAY)
# normalized_iris = cv2.equalizeHist(normalized_iris)
# segmented_iris = np.expand_dims(normalized_iris, axis=0)  # Add batch dimension
# predict = model.predict(segmented_iris)
# print(np.argmax(predict.flatten()))


from keras.applications.vgg19 import VGG19
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models       import Model
from keras.applications.densenet import DenseNet201

model = load_model("iris_recognition_model.h5")


img_path = "F:\\repo\\iris_rec\\normalized_casia\\45.5.jpg"
im = cv2.imread(img_path, 0)

# #  segmentation
# segmented_iris, cirpupil, ciriris = segment(im)

# segmented_iris = cv2.resize(segmented_iris, (224, 224))  # Resize to match Densenet input
segmented_iris = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
# segmented_iris = img_to_array(segmented_iris)
# segmented_iris = np.expand_dims(segmented_iris, axis=0)  # Add batch dimension
# segmented_iris = preprocess_input(segmented_iris)

#segmented_iris = cv2.resize(segmented_iris, (124, 124))
#normalized_iris = cv2.equalizeHist(segmented_iris)


# # #  feature encoding
#encoded_iris = feature_encoding(normalized_iris)
normalized_iris = image.img_to_array(segmented_iris)
normalized_iris = np.expand_dims(normalized_iris, axis=0)  # Add batch dimension
print(normalized_iris.shape)
features = model.predict(normalized_iris)
print(features)
print(features.shape)
print(np.argmax(features.flatten()))