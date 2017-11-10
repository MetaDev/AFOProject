import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import scipy
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, normalize
import data_prep
import data_relevance

from sklearn import linear_model

def evaluate_by_example(X_pre, Y, data_prep_scalerY,model, length=0.1):

    #predict with the strain values, calculate difference and convert to degrees
    pred_test_Y = model.predict(X_pre)
    #denormalise data
    pred_test_Y=data_prep_scalerY.inverse_transform(pred_test_Y)
    Y=data_prep_scalerY.inverse_transform(Y)
    #calculate the error given predicted data
    degree_error= [(calc_angle_degree(pr_y,length)-calc_angle_degree((t_y),length)) for pr_y,t_y in zip(pred_test_Y,Y)]
    #return list of avtual deformation angle in degrees and it's corresponding error
    return list(zip([calc_angle_degree((t_y),length) for t_y in Y],[e for e in degree_error]))

# in my test setup the displacement is in dm and length in mm, thus I have to convert the length to dm by dividing by 100
# the angle is calculated using the by the cosine rule in straight angle triangles
def calc_angle_degree(displacement,length):
    #the first ordinal is x, the second y
    return np.rad2deg(np.tanh(displacement/length))

# def visualise_accuracy_GP(X_raw,Y_raw):

#normalise and flatten data
X,Y,_,scaler_Y= data_prep.preprocessing_data(X,Y,sensor_axis =[0])

n_max=10

#extract most significant sensors, and random sensors
n_sensors=10
X_ind_rel=data_relevance.calc_most_significant_sensor(X,Y,n_sensors=n_sensors,test_method=2)
X_ind_rand= np.random.choice(len(X[0]), n_sensors, replace=False)

X_rel=X[:,X_ind_rel]
X_rand=X[:,X_ind_rand]
X=X_rel

X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.33, random_state=42)

#pick a different layout for each trial for angle evaluation
# to average out differences in results of the model, NN trains stochastically
model = gp

# Fit the model
model.fit(X_train, Y_train)
# evaluate the model

degree_test=evaluate_by_example(X_test, Y_test,scaler_Y,model)
#plot heatmap, average over all n sensors

#make the degree error format: angleX,angleY,errorX,errorY
degree_test=np.array(degree_test).reshape(len(degree_test),-1)

from scipy.interpolate import griddata
# grid the data.
# define grid.
xi = np.linspace(-60,60,100)
yi = np.linspace(-60,60,100)

zi = griddata(( degree_test[:,0],  degree_test[:,1]), np.abs(degree_test[:,2]+degree_test[:,3]),(xi[None,:], yi[:,None]), method='linear')
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
#plt.savefig(r"C:\Users\Administrator\Google Drive\Windows\Research\Project\FEM\Results\learning\visual\heatmap_error_rectangular.png")
plt.show()
