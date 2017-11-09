import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
def build_simple_nn(X):
    n_input = len(X[0])
    # create model
    model = Sequential()
    model.add(Dense(n_input, input_dim=n_input,
                    kernel_initializer='normal', activation='relu'))
    # adding more than two layer actually lower accuracy, while it should be overfitting, this could be due to decreased compatibility with other hyperparameters
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #2d displacement
    model.add(Dense(2, kernel_initializer='normal'))
    adam = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.02, patience=10, verbose=0, mode='auto')
    # Compile model
    model.compile(loss='mse', optimizer=adam, metrics=['mse', 'accuracy'])
    return model
from sklearn.gaussian_process import GaussianProcessRegressor

import sklearn.model_selection as ms
def test_GP(X,Y):
    gp = GaussianProcessRegressor( n_restarts_optimizer=9)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.33, random_state=42)

    # Fit the model
    gp.fit(X_train, Y_train)
    # evaluate the model
    y_mean, y_cov = gp.predict(X_test, return_cov=True)
    y_error = np.mean((y_mean-Y_test)**2)
    return y_error