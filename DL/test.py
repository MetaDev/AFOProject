from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import sklearn.model_selection

#read input
data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
x_file="\\strain_list.npy"
y_file="\\displacement_list.npy"
X = np.load(data_path+x_file)
print(X)
Y = np.load(data_path+y_file)
print(Y)
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

