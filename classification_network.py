from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

from keras import optimizers
import keras

def classificator(input_shape, num_classes):

    model = Sequential()
    model.add(Flatten(input_shape=input_shape[1:])) 
    model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.4)) 
    model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes, activation="softmax"))
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    return model