
#old visualisation version check GP version for newer code
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
import data_prep
import data_relevance
disp_node_strain,disp,nodes_i_coord=data_prep.read_data()
#normalise and flatten data
X,Y,_,_= data_prep.preprocessing_data(disp_node_strain,disp,sensor_axis =[0])

n_sensors=10
X_ind_rel=data_relevance.calc_most_significant_sensor(X,Y,n_sensors=n_sensors,test_method=2)
X_ind_rand= np.random.choice(len(X[0]), n_sensors, replace=False)

X_rel=X[:,X_ind_rel]
X_rand=X[:,X_ind_rand]

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
      
        # to average out differences in results of the model, NN trains stochastically
        model = build_model(X_rel)

        # fix random seed for reproducibility
        seed = 7
        kfold = ms.KFold(n_splits=5, random_state=seed)
        cvscores_mse = []
        cvscores_val_acc = []
        for train, test in kfold.split(X_rel, Y):
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
zi = griddata(( test_mean[:,0],  test_mean[:,1]), test_mean[:,2]+test_mean[:,3],(xi[None,:], yi[:,None]), method='linear')
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
