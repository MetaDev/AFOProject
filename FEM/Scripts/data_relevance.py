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
    return np.array(sorted(enumerate(score_y0+score_y1), key=lambda x: x[1], reverse=True)[0:n_sensors]).astype(int)[:,0]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import sklearn.model_selection as ms

def test_relevance_selection():
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
    print("kak")
    gp = GaussianProcessRegressor( n_restarts_optimizer=9)
    seed = 7
    kfold = ms.KFold(n_splits=4, random_state=seed)

    #compare mse with relevant sensors and with random ones
    #TODO lise the position of these sensors
    test_GP(X_rand,Y)
    test_GP(X_rel,Y)
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from pylab import *
def cluster_nodes(nodes_i_coord):
    node_pos=[nodes_i_coord[key] for key in sorted(nodes_i_coord.keys())]
    n_clusters=200
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_pos)
    node_clusters_coord=[[] for i in range(n_clusters)]
    node_clusters_i=[[] for i in range(n_clusters)]
    for i,(cluster, node) in enumerate(zip(kmeans.labels_,node_pos)):
        node_clusters_coord[cluster].append(node)
        node_clusters_i[cluster].append(i)
    return node_clusters_i,node_clusters_coord
def test_GP(X,Y):
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.33, random_state=42)

    # Fit the model
    gp.fit(X_train, Y_train)
    # evaluate the model
    y_mean, y_cov = gp.predict(X_test, return_cov=True)
    y_error = np.mean((y_mean-Y_test)**2)
    print(y_error)  
def visualise_cluster(coords,colors=None,values=None, view_angle=15):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colmap = cm.ScalarMappable(cmap=cm.viridis)
   
    colmap.set_array(values)

    for i,node_coords in enumerate(node_clusters_coord):
        node_coords = np.array(node_coords).T
        if colors is not None:
            s=ax.scatter(node_coords[0], node_coords[1], node_coords[2], c=colors[i],s=100, marker='o')
        if values is not None:
            s=ax.scatter(node_coords[0], node_coords[1], node_coords[2], c=cm.viridis(values[i]),s=100, marker='o')
        #turn off opacity relative to camera distance
        s.set_edgecolors = s.set_facecolors = lambda *args:None
    ax.set_aspect('equal', adjustable='box')
    if values is not None:
        fig.colorbar(colmap)
    ax.view_init(elev=10, azim=view_angle)
    plt.show()

def visualise_sample_relevance():
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

    #to respect the index of the nodes in the data I sort on their node index
    node_pos=[nodes_i_coord[key] for key in sorted(nodes_i_coord.keys())]
    n_clusters=200
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_pos)
    node_clusters_coord=[[] for i in range(n_clusters)]
    node_clusters_i=[[] for i in range(n_clusters)]

    def get_cluster_center(center,node_coords,node_i,n_nodes=1):
        return (sorted(zip(node_i,node_coords), key=lambda node_i : np.linalg.norm(node_i[1]-center)))[0:n_nodes]

    for i,(cluster, node) in enumerate(zip(kmeans.labels_,node_pos)):
        node_clusters_coord[cluster].append(node)
        node_clusters_i[cluster].append(i)
    test=cm.rainbow(np.arange(n_clusters))
    visualise_cluster(node_clusters_coord,test)

    #sample random 1 node from each cluster and train on each cluster left out
    #TODO alternative is to take the center of the cluster
    center_cluster_inds = [get_cluster_center(center,node_coord,node_i)[:,0]
            for center,node_coord,node_i in zip(kmeans.cluster_centers_,node_clusters_coord,node_clusters_i)]
    rand_cluster_ints = [np.random.choice(i_s,replace=False) for i_s in node_cluster_i]
    print(center_cluster_inds)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X[:,center_cluster_inds], Y, test_size=0.33, random_state=42)


    gp = GaussianProcessRegressor( n_restarts_optimizer=9)
    # Fit the model
    gp.fit(X_train, Y_train)
    # evaluate the model
    y_mean, y_cov = gp.predict(X_test, return_cov=True)
    y_error = np.mean((y_mean-Y_test)**2)
    print(y_error)
    #training on all sensor values is ill composed (1000 variables 1000 data points)
    #thus train on the center nodes of the clusters

    error_diff=[]
    for omit_sensor in range(len(rand_cluster_ints)):
        X_test_omit=X_test.copy()
        X_test_omit[:,omit_sensor]=np.ones((len(X_test_omit)))
        y_mean_omit, y_cov = gp.predict(X_test_omit, return_cov=True)
        
        y_error_omit = np.mean((y_mean_omit-Y_test)**2)
        error_diff.append(y_error_omit-y_error)


    # error_diff=np.array(error_diff)**4
    #normalise error diff
    values=[(error/max(error_diff)) for error in error_diff]

    visualise_cluster(node_clusters_coord,values=values)
    visualise_cluster(node_clusters_coord,values=values,view_angle=135)

visualise_sample_relevance()

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

#to respect the index of the nodes in the data I sort on their node index
node_pos=[nodes_i_coord[key] for key in sorted(nodes_i_coord.keys())]
n_clusters=50
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_pos)
node_clusters_coord=[[] for i in range(n_clusters)]
node_clusters_i=[[] for i in range(n_clusters)]
def get_cluster_center(center,node_coords,node_i,n_nodes=2):
    return (sorted(zip(node_i,node_coords), key=lambda node_i : np.linalg.norm(node_i[1]-center)))[0:n_nodes]

for i,(cluster, node) in enumerate(zip(kmeans.labels_,node_pos)):
    node_clusters_coord[cluster].append(node)
    node_clusters_i[cluster].append(i)
test=cm.rainbow(np.arange(n_clusters))
visualise_cluster(node_clusters_coord,test)

#sample random 1 node from each cluster and train on each cluster left out
#TODO alternative is to take the center of the cluster
center_cluster_inds = [list(zip(*get_cluster_center(center,node_coord,node_i)))[0]
        for center,node_coord,node_i in zip(kmeans.cluster_centers_,node_clusters_coord,node_clusters_i)]
#use one of both indices
rand_cluster_ints = [np.random.choice(i_s,replace=False) for i_s in node_clusters_i]
X_train, X_test, Y_train, Y_test = ms.train_test_split(np.reshape(X[:,rand_cluster_ints],(len(X),-1))
                                                                        , Y, test_size=0.33, random_state=42)


print( np.shape(X_train))
print( np.shape(Y_train))
gp = GaussianProcessRegressor( n_restarts_optimizer=9)
# Fit the model
gp.fit(X_train, Y_train)
# evaluate the model
y_mean, y_cov = gp.predict(X_test, return_cov=True)
y_error = np.mean((y_mean-Y_test)**2)
print(y_error)
#training on all sensor values is ill composed (1000 variables 1000 data points)
#thus train on the center nodes of the clusters
import train_model
model=train_model.build_simple_nn(X_train)
model.fit(X_train,Y_train)

error_diff=[]
for omit_sensor in range(len(rand_cluster_ints)):
    X_test_omit=X_test.copy()
    X_test_omit[:,omit_sensor]=np.zeros((len(X_test_omit)))
    y_mean_omit, y_cov = gp.predict(X_test_omit, return_cov=True)
    # y_mean_omit = model.predict(X_test_omit)
    y_error_omit = np.mean((y_mean_omit-Y_test)**2)
    error_diff.append(y_error_omit-y_error)

# error_diff=np.array(error_diff)**4
#normalise error diff
values=[(error/max(error_diff))**8 for error in error_diff]
visualise_cluster(node_clusters_coord,values=values)
visualise_cluster(node_clusters_coord,values=values,view_angle=135)

# test_relevance_selection()

#TODO train on each cluster seperately and check accuracy

