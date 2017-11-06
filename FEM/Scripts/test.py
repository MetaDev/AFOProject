import sklearn.feature_selection
import matplotlib.pyplot as plt
import numpy as np
import data_prep

from scipy import stats
import numpy as np

def calc_most_significant_sensor(X,Y,n_sensors,test_method=0):
    if test_method==0:
        score_y0=sklearn.feature_selection.mutual_info_regression(X, Y[:,0],
                                            discrete_features =False, n_neighbors=3, copy=True, random_state=1)
        score_y1=sklearn.feature_selection.mutual_info_regression(X, Y[:,1],
                                            discrete_features =False, n_neighbors=3, copy=True, random_state=1)
    #F values of features. p-values of F-scores.
    #http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
    elif test_method==1:
        score_y0=sklearn.feature_selection.f_regression(X, Y[:,0], center=True)[0]
        score_y1=sklearn.feature_selection.f_regression(X, Y[:,1], center=True)[0]
    #1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation
    elif test_method==2:
        score_y0=np.abs(np.array([stats.pearsonr(x_col,Y[:,0]) for x_col in X.T])[:,0])
        score_y1=np.abs(np.array([stats.pearsonr(x_col,Y[:,1]) for x_col in X.T])[:,0])
    #return index of max scores
    return np.array(sorted(enumerate(score_y0+score_y1), key=lambda x: x[1], reverse=True)[0:n_max]).astype(int)[:,0]

disp_node_strain,disp,nodes_i_coord=data_prep.read_data()
#normalise and flatten data
X,Y,_,_= data_prep.preprocessing_data(disp_node_strain,disp,sensor_axis =[0])

n_max=10

#extract most significant sensors, and random sensors
n_sensors=10
X_ind_rel=calc_most_significant_sensor(X,Y,n_sensors=n_sensors,test_method=2)
X_ind_rand= np.random.choice(len(X[0]), n_sensors, replace=False)

X_rel=X[:,X_ind_rel]
X_rand=X[:,X_ind_rand]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

gp = GaussianProcessRegressor( n_restarts_optimizer=9)
import train_model
model = train_model.build_simple_nn(X_rel)
import sklearn.model_selection as ms
seed = 7
kfold = ms.KFold(n_splits=3, random_state=seed)
def train(X):
    mse=[]
    for train, test in kfold.split(X, Y):
        # Fit the model
        gp.fit(X[train], Y[train])
        # evaluate the model
        y_mean, y_cov = gp.predict(X[test], return_cov=True)
        y_error = np.mean((y_mean-Y[test])**2)
    print(np.mean(mse),np.std(mse))
#compare mse with relevant sensors and with random ones
#todo visulise the position of these sensors
# train(X_rand)
# train(X_rel)

#visualise relevantness as heatmap
#train GP on all sensors 
# predict with deleted sensor values and non deleted
#instead of removing sensors one by one, might be slow, cluster the points and remove cluster, visulise cluster using pyplot
# 
from sklearn.cluster import KMeans
#to respect the index of the nodes in the data I sort on their node index
node_pos=[nodes_i_coord[key] for key in sorted(nodes_i_coord.keys())]
n_clusters=50
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_pos)
node_clusters_coord=[[] for i in range(n_clusters)]
node_cluster_i=[[] for i in range(n_clusters)]
for i,(cluster, node) in enumerate(zip(kmeans.labels_,node_pos)):
    node_clusters_coord[cluster].append(node)
    node_cluster_i[cluster].append(i)
#visualise clusters
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

traces=[]


import colorsys
N = n_clusters
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for i,node_coords in enumerate(node_clusters_coord):
    node_coords = np.array(node_coords).T
    ax.scatter(node_coords[0], node_coords[1], node_coords[2], c=RGB_tuples[i], marker='o')
ax.set_aspect('equal', adjustable='box')
plt.show()


#todo debug error
cluster_mse=[]
for omit_sensor in node_cluster_i:
    error_diff=[]
    for train, test in kfold.split(X, Y):
        # Fit the model
        gp.fit(X[train], Y[train])
        # evaluate the model
        X_test_omit=X[test]
        X_test_omit[:,omit_sensor]=np.zeros((len(X_test_omit),len(omit_sensor)))
        y_mean_omit, y_cov = gp.predict(X_test_omit, return_cov=True)
        y_mean, y_cov = gp.predict(X[test], return_cov=True)
        y_error_omit = np.mean((y_mean_omit-Y[test])**2)
        y_error = np.mean((y_mean-Y[test])**2)
        error_diff.append(y_error-y_error_omit)
    #print(np.mean(mse),np.std(mse))
    cluster_mse.append(np.mean(error_diff))
from pylab import *
colmap = cm.ScalarMappable(cmap=cm.hsv)
colmap.set_array(cluster_mse)
for node_coords,mse in zip(node_clusters_coord,cluster_mse):
    node_coords = np.array(node_coords).T
    ax.scatter(node_coords[0], node_coords[1], node_coords[2], c=cm.hsv(mse/max(cluster_mse)), marker='o')
ax.set_aspect('equal', adjustable='box')
cb = fig.colorbar(colmap)
plt.show()
