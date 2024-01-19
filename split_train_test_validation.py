import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data_to_fit(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test