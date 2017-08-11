import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import normalize
import vector_calc as vc

#read input
data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
data_type=2
if data_type==0:
	x_file="\\all_strain_list"
elif data_type==1:
	x_file="\\surface_strain_list"
else:
	x_file="\\proj_surface_strain_list"
y_file="\\displacement_list"

X = np.load(data_path+x_file+".npy")
Y = np.load(data_path+y_file+".npy")

def preprocessing_data(X,Y,n_sensors):
	print("n_sensors: "+ str(n_sensors))
	#TODO extract random sensor not just the first 
	#extract n_sensor columns
	X=np.delete(X, tuple(range(len(X[0])-n_sensors)), axis=1)
#extract x axis value
	X=X[:,:,[0]]

	#flatten X
	X=X.reshape(len(Y),-1)
	#scale to mean 0 and unit variance
	X=scale(X)

	Y=scale(Y)
	return X,Y

def build_model(X):
	n_input = len(X[0])
	# create model
	model = Sequential()
	model.add(Dense(n_input, input_dim=n_input, kernel_initializer='normal', activation='relu'))
	model.add(Dense(32))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(32))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(3, kernel_initializer='normal'))
	adam=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=10, verbose=0, mode='auto')
	# Compile model
	model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])
	return model

for n_sensors in range(1,15):
	
	X_pre,Y_pre = preprocessing_data(X,Y,n_sensors)
	
	model = build_model(X_pre)
	# fix random seed for reproducibility
	seed = 7
	kfold = ms.KFold(n_splits=5, random_state=seed)
	cvscores_mse = []
	cvscores_val_acc = []
	for train, test in kfold.split(X_pre, Y_pre):
		# Fit the model
		history = model.fit(X_pre[train], Y_pre[train],validation_split=0.1, epochs=10, 
											batch_size=5, verbose=0)
		# evaluate the model
		scores = model.evaluate(X_pre[test], Y_pre[test], verbose=0)
		#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores_val_acc.append(scores[2] * 100)
		cvscores_mse.append(scores[1] * 100)
	# print("mse")
	# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_mse), np.std(cvscores_mse)))
	# print("accuracy")
	# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_val_acc), np.std(cvscores_val_acc)))
	print(np.mean(cvscores_val_acc))

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


