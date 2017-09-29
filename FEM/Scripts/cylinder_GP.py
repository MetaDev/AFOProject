import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import xlwings as xw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, normalize
import scipy

from sklearn import linear_model
import vector_calc as vc

# read input
data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
result_path = r'C:\Users\Administrator\Google Drive\Windows\Research\Project\Docs\simple_AFO_results.xlsx'
data_type = 2
n_sensor_axis = 1
axis_training = {1: [0], 2: [0, 1], 3: [0, 1, 2]}
# name_afo_project = "_cylinder"
name_afo_project = "_rot_cube"
# name_afo_project = "_simple_afo_big"
#the length (in dm) from the deformed object is used to calculate the angle based on displacement
length=0.1
if data_type == 0:
    x_file = "\\all_strain_list_simple" + name_afo_project
elif data_type == 1:
    x_file = "\\surface_strain_list" + name_afo_project
else:
    x_file = "\\proj_surface_strain_list" + name_afo_project
disp_vectors = "\\displacement_list"+ name_afo_project
#in mm
disp_coord = "\\displacement_coord"+ name_afo_project

X = np.load(data_path + x_file + ".npy")
Y = np.load(data_path + disp_vectors + ".npy")
Y_coord = np.load(data_path + disp_coord + ".npy")
#two options to interpolate 1, weighted sum of X closest points, interpolate function over all values with barycentric coordinates
# Y_func_1 =  lambda x, y : x + y
#2
#the ranges to train the model on: x=0||2 , y=[0,5], z= [0,10]

#a different function for each displacement
Y_funcs=[ scipy.interpolate.LinearNDInterpolator(points=Y_coord,values=X[i]) for i in range(len(X))]

print(Y_funcs[0](0,2,3))
# for i in range(3):
#     plt.hist(X[:,0,i], bins='auto')  # arguments are passed to np.histogram
#     plt.show()
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
def evaluate_by_example(X_pre, Y,model, length):
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
    degree_error= [(calc_angle_degree(pr_y,length)-calc_angle_degree((t_y),length)) for pr_y,t_y in zip(pred_test_Y,test_Y)]
    #return list of avtual deformation angle in degrees and it's corresponding error
    return list(zip([calc_angle_degree((t_y),length) for t_y in test_Y],[e for e in degree_error]))

    # in my test setup the displacement is in dm and length in mm, thus I have to convert the length to dm by dividing by 100
def calc_angle_degree(displacement,length):
    #the first ordinal is x, the second y
    return np.rad2deg(np.tanh(displacement/length))

# Instanciate a Gaussian Process model
kernel = 1.0 * RBF() \
    + WhiteKernel()

gp = GaussianProcessRegressor(kernel=kernel,alpha=0, n_restarts_optimizer=9)
# # Initialize model.
# logreg = linear_model.LinearRegression()
# X_pre, Y_pre = preprocessing_data(X, Y,10)
# # # Use cross_val_score to automatically split, fit, and score.
# scores = ms.cross_val_score(gp, X_pre, Y_pre, cv=10,scoring="r2")
# print(scores)
n_trials = 1
cv_results = []
degree_test= []
for i, n_s in enumerate(range(10, 11)):
    # for evaluation of a model accuracy the sensor layout should be kept constant over trials 
    # X_pre, Y_pre = preprocessing_data(X, Y, n_s)

    mse=[]
    for n in range(n_trials):
        #pick a different layout for each trial for angle evaluation
        X_pre, Y_pre = preprocessing_data(X,Y,n_s)
        # to average out differences in results of the model, NN trains stochastically
        model = gp
        # fix random seed for reproducibility
        seed = 7
        kfold = ms.KFold(n_splits=5, random_state=seed)
        cv_score_mse = []

        for train, test in kfold.split(X_pre, Y_pre):
            # Fit the model
            model.fit(X_pre[train], Y_pre[train])
            # evaluate the model

            y_mean, y_cov = gp.predict(X_pre[test], return_cov=True)
            y_error = np.mean((y_mean-Y_pre[test])**2)
            mse.append(y_error * 100)
            
        # degree_test.append(evaluate_by_example(X_pre, Y,model))
        cv_score_mse.append([np.mean(mse),np.std(mse)])
        
    #calc mean and std from degree test
    degree_test.append(evaluate_by_example(X_pre, Y,model,length))
    cv_results.append([n_s, np.mean(np.array(cv_score_mse)[:,0]),
    np.mean(np.array(cv_score_mse)[:,1])])
print(cv_results)
#plot heatmap
test_mean=np.mean(degree_test,axis=0).reshape(len(degree_test[0]),-1)

A =np.stack((
     test_mean[:,0],
     test_mean[:,1],
     test_mean[:,2]+test_mean[:,3]),axis=1)
from scipy.interpolate import griddata
# grid the data.
# define grid.
xi = np.linspace(-60,60,100)
yi = np.linspace(-60,60,100)
zi = griddata(( test_mean[:,0],  test_mean[:,1]), test_mean[:,2]+test_mean[:,3],(xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
#plt.savefig(r"C:\Users\Administrator\Google Drive\Windows\Research\Project\FEM\Results\learning\visual\heatmap_error_rectangular.png")
plt.show()
