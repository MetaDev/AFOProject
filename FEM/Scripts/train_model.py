import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
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