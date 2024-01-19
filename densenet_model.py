from keras.models       import Model
from keras.applications.densenet import DenseNet201

def densenet_model():
    model = DenseNet201(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    model.summary()

    return model