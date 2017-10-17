import sklearn.feature_selection
import matplotlib.pyplot as plt
import numpy as np
import data_prep

from scipy import stats
import numpy as np


disp_node_strain,disp=data_prep.read_data()
X,Y,_,_= data_prep.preprocessing_data(disp_node_strain,disp,sensor_axis =[0])

n_max=10
sklearn.feature_selection.mutual_info_regression
def calc_most_significant_sensor(X,Y,n_sensors,test_method="mutual_info"):
    if test_method=="mutual_info":
        score_y0=sklearn.feature_selection.mutual_info_regression(X, Y[:,0],
                                            discrete_features =False, n_neighbors=3, copy=True, random_state=1)
        score_y1=sklearn.feature_selection.mutual_info_regression(X, Y[:,1],
                                            discrete_features =False, n_neighbors=3, copy=True, random_state=1)
    #F values of features. p-values of F-scores.
    #http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
    elif test_method=="f_regression":
        score_y0=sklearn.feature_selection.f_regression(X, Y[:,0], center=True)
        score_y1=sklearn.feature_selection.f_regression(X, Y[:,1], center=True)
    #1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation
    elif test_method=="pearson":
        score_y0=np.abs(np.array([stats.pearsonr(x_col,Y[:,0]) for x_col in X.T])[:,0])
        score_y1=np.abs(np.array([stats.pearsonr(x_col,Y[:,1]) for x_col in X.T])[:,0])
    #return index of max scores
    return np.array(sorted(enumerate(score_y0+score_y1), key=lambda x: x[1], reverse=True)[0:n_max])[:,0]

#extract most significant sensors, and random sensors
X_ind=calc_most_significant_sensor(X,Y,n_sensors=10)
#todo


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
kernel = 1.0 * RBF() \
    + WhiteKernel()

gp = GaussianProcessRegressor(kernel=kernel,alpha=0, n_restarts_optimizer=9)
import sklearn.model_selection as ms
seed = 7
kfold = ms.KFold(n_splits=5, random_state=seed)
# for train, test in kfold.split(X_pre, Y_pre):
#     # Fit the model
#     model.fit(X_pre[train], Y_pre[train])
#     # evaluate the model

#     y_mean, y_cov = gp.predict(X_pre[test], return_cov=True)
#     y_error = np.mean((y_mean-Y_pre[test])**2)
#     mse.append(y_error * 100)

#visualise found relevant sensors
