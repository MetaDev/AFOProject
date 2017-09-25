import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import vector_calc as vc
import xlwings as xw


# read input
data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
result_path = r'C:\Users\Administrator\Google Drive\Windows\Research\Project\Docs\simple_AFO_results.xlsx'
data_type = 2
n_sensor_axis = 1
axis_training = {1: [0], 2: [0, 1], 3: [0, 1, 2]}
name_afo_project = "_cylinder"
if data_type == 0:
    x_file = "\\all_strain_list_simple" + name_afo_project
elif data_type == 1:
    x_file = "\\surface_strain_list" + name_afo_project
else:
    x_file = "\\proj_surface_strain_list" + name_afo_project
y_file = "\\displacement_list"+ name_afo_project

X = np.load(data_path + x_file + ".npy")
Y = np.load(data_path + y_file + ".npy")
scalerX = StandardScaler()
scalerY = StandardScaler()
def preprocessing_data(X, Y, n_sensors):
    # extract random n_sensor columns
    X = np.delete(X, (tuple(np.random.choice(len(X[0]), len(X[0]) - n_sensors, replace=False))), axis=1)
    
    # extract x axis value
    X = X[:, :, axis_training[n_sensor_axis]]
    
    # flatten X
    X = X.reshape(len(Y), -1)
    
    # scale to mean 0 and unit variance
    X = scalerX.fit_transform(X)
    noise=np.random.normal(0,0.1,np.shape(X))
    X=X+noise
    Y = scalerY.fit_transform(Y)
    return X, Y


def build_model(X):
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

def evaluate_by_example(X_pre, Y,model):
    # #pick strain values from X, with a label from -40 to 20
    # to_pred_points = np.linspace(-4,2,7)

    # #find closest x axis displacement in label list
    # test_data_indexed = ([min(enumerate(Y), key=lambda x: abs(x[1][0]-pr)) for pr in to_pred_points])
    test_data_indexed=list(enumerate(Y))
    #predict with the strain values, calculate difference and convert to degrees
    pred_test_Y = model.predict(X_pre[np.array([id[0] for id in test_data_indexed])])
    #denormalise data
    pred_test_Y=scalerY.inverse_transform(pred_test_Y)
    test_Y=np.array([id[1] for id in test_data_indexed])
    #calculate the error given predicted data
    degree_error= [(calc_angle_degree(pr_y,4)-calc_angle_degree((t_y),4)) for pr_y,t_y in zip(pred_test_Y,test_Y)]
    #return list of avtual deformation angle in degrees and it's corresponding error
    return list(zip([calc_angle_degree((t_y),4) for t_y in test_Y],[e for e in degree_error]))

    # the length and dsiplacment should both have the same unit
    # in my test setup the displacement is in dm and length in mm
def calc_angle_degree(displacement,length):
    #the first ordinal is x, the second y
    return np.rad2deg(np.tanh(displacement/length))

n_trials = 5
xl_arr = []
degree_test= []
xl_degree_test= []
for i, n_s in enumerate(range(10, 11)):
    # for evaluation of a model accuracy the sensor layout should be kept constant over trials 
    # X_pre, Y_pre = preprocessing_data(X, Y, n_s)

    cv_score_acc_mean = []
    cv_score_acc_std = []
    for n in range(n_trials):
        #pick a different layout for each trial for angle evaluation
        X_pre, Y_pre = preprocessing_data(X, Y, n_s)
        # to average out differences in results of the model, NN trains stochastically
        model = build_model(X_pre)

        # fix random seed for reproducibility
        seed = 7
        kfold = ms.KFold(n_splits=5, random_state=seed)
        cvscores_mse = []
        cvscores_val_acc = []
        for train, test in kfold.split(X_pre, Y_pre):
            # Fit the model
            history = model.fit(X_pre[train], Y_pre[train], epochs=100,
                                batch_size=10, verbose=0)
            
            # evaluate the model
            scores = model.evaluate(X_pre[test], Y_pre[test], verbose=0)
            #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores_val_acc.append(scores[2] * 100)
            cvscores_mse.append(scores[1] * 100)
            
        # print("mse")
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_mse), np.std(cvscores_mse)))
        # print("accuracy")
        degree_test.append(evaluate_by_example(X_pre, Y,model))
        cv_score_acc_mean.append(np.mean(cvscores_val_acc))
        cv_score_acc_std.append(np.std(cvscores_val_acc))
    #calc mean and std from degree test
    xl_arr.append([n_s, np.mean(cv_score_acc_mean),
                   np.mean(cv_score_acc_std)])
print(np.mean(degree_test,axis=0).reshape(len(degree_test[0]),-1))
print(np.std(degree_test,axis=0).reshape(len(degree_test[0]),-1)[:,2:4])

#plot heatmap
test_mean=np.mean(degree_test,axis=0).reshape(len(degree_test[0]),-1)

A =np.stack((
     test_mean[:,0],
     test_mean[:,1],
     test_mean[:,2]+test_mean[:,3]),axis=1)

from scipy.interpolate import griddata
# grid the data.
# define grid.
xi = np.linspace(-45,25,100)
yi = np.linspace(-45,25,100)
zi = griddata(( test_mean[:,0],  test_mean[:,1]), test_mean[:,2]+test_mean[:,3],(xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
# plt.scatter(test_mean[:,0],test_mean[:,1],marker='o',c='b',s=5)
plt.savefig("heatmap_degree_error.png")

plt.show()

# save results in excel
# import xlwings as xw
# wb = xw.Book(result_path)  # this will create a new workbook
# import datetime
# sht = wb.sheets.add(datetime.datetime.now().strftime(
#     '%m-%d-%h-%M') + " axisdim-" + str(n_sensor_axis))
# sht.range('A1').value = np.concatenate( 
#     (np.mean(degree_test,axis=0).reshape(7,-1),
#     np.std(degree_test,axis=0).reshape(7,-1)[:,2:4]),axis=1)

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
