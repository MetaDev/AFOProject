import sklearn.feature_selection
import matplotlib.pyplot as plt
import numpy as np
import data_prep

from scipy import stats
import numpy as np
import itertools

def calc_most_significant_sensor(X,Y,test_method=0):
    scores=[]
    for x,y in itertools.product([0,1,2],[0,1]):
        X_score=np.reshape(X[:,:,x],(len(X),-1))
        Y_score=Y[:,y]
        if test_method==0:
            score=sklearn.feature_selection.mutual_info_regression(X_score, Y_score,
                                                discrete_features =False, n_neighbors=3, copy=True, random_state=1)
        #F values of features. p-values of F-scores.
        #http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
        elif test_method==1:
            score=sklearn.feature_selection.f_regression(X_score, Y_score, center=True)[0]
        #1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation
        elif test_method==2:
            score=np.abs(np.array([stats.pearsonr(x_col,Y_score) for x_col in X_score.T])[:,0])
        scores.append(score)
    scores=np.mean(scores,axis=0)
    #return index of max scores
    return np.array(sorted(enumerate(scores), key=lambda x: x[1], reverse=True)).astype(int)[:,0]

from sklearn.gaussian_process import GaussianProcessRegressor

import sklearn.model_selection as ms


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
    gp = GaussianProcessRegressor( n_restarts_optimizer=9)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.33, random_state=42)

    # Fit the model
    gp.fit(X_train, Y_train)
    # evaluate the model
    y_mean, y_cov = gp.predict(X_test, return_cov=True)
    y_error = np.mean((y_mean-Y_test)**2)
    return y_error
def visualise_coords(coords,colors,size=100,view_angle=15):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coords = np.array(coords).T
    
    s=ax.scatter(coords[0], coords[1], coords[2], c=colors,s=size, marker='o')
       
    #turn off opacity relative to camera distance
    s.set_edgecolors = s.set_facecolors = lambda *args:None
    ax.set_aspect('equal', adjustable='box')
    ax.view_init(elev=10, azim=view_angle)
    plt.show()
def visualise_cluster(node_clusters_coord,colors=None,values=None, view_angle=15,size=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colmap = cm.ScalarMappable(cmap=cm.viridis)
   
    colmap.set_array(values)

    for i,node_coords in enumerate(node_clusters_coord):
        node_coords = np.array(node_coords).T
        if colors is not None:
            s=ax.scatter(node_coords[0], node_coords[1], node_coords[2], c=colors[i],s=size, marker='o')
        if values is not None:
            s=ax.scatter(node_coords[0], node_coords[1], node_coords[2], c=cm.viridis(values[i]),s=size, marker='o')
        #turn off opacity relative to camera distance
        s.set_edgecolors = s.set_facecolors = lambda *args:None
    ax.set_aspect('equal', adjustable='box')
    if values is not None:
        fig.colorbar(colmap)
    ax.view_init(elev=10, azim=view_angle)
    plt.show()
#returns the nodes closest to the cluster center
def get_closest_cluster_center(center,node_coords,node_i):
        return (sorted(zip(node_i,node_coords), key=lambda node_i : np.linalg.norm(node_i[1]-center)))
def get_3D_clusters(node_coords,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_coords)
    node_clusters_coord=[[] for i in range(n_clusters)]
    node_clusters_i=[[] for i in range(n_clusters)]
    for i,(cluster, node) in enumerate(zip(kmeans.labels_,node_coords)):
        node_clusters_coord[cluster].append(node)
        node_clusters_i[cluster].append(i)
    return node_clusters_i,node_clusters_coord,kmeans.cluster_centers_
# ##1
# # test the accuracy difference between relevant strains and random
# disp_node_strain, disp, nodes_i_coord = data_prep.read_data()
# #normalise and flatten data
# X = disp_node_strain
# Y = disp
# #extract most significant sensors, and random sensors
# n_sensors = 10
# X_ind_rel = calc_most_significant_sensor(
#     X, Y,  test_method=2)[0:n_sensors]
# X_ind_rand = np.random.choice(len(X[0]), n_sensors, replace=False)

# X_rel = X[:, X_ind_rel, :]
# X_rand = X[:, X_ind_rand, :]
# X_rand, Y, _, _ = data_prep.preprocessing_data(X_rand, Y, sensor_axis=[0])
# X_rel, Y, _, _ = data_prep.preprocessing_data(X_rel, Y, sensor_axis=[0])

# #compare mse with relevant sensors and with random ones
# #TODO list the position of these sensors
# print(test_GP(X_rand, Y))
# print(test_GP(X_rel, Y))

# ##2
# #visualise the influence of leaving out clusters of nodes on the prediction

# disp_node_strain,disp,nodes_i_coord=data_prep.read_data()
# #normalise and flatten data, TODO the flattening should happen after the sensor selection
# X,Y,_,_= data_prep.preprocessing_data(disp_node_strain,disp,sensor_axis =[0])

# n_max=10

# #extract most significant sensors, and random sensors
# n_sensors=10
# X_ind_rel=calc_most_significant_sensor(X,Y,test_method=2)[0:n_sensors]
# X_ind_rand= np.random.choice(len(X[0]), n_sensors, replace=False)

# X_rel=X[:,X_ind_rel]
# X_rand=X[:,X_ind_rand]

# #to respect the index of the nodes in the data I sort on their node index
# node_pos=[nodes_i_coord[key] for key in sorted(nodes_i_coord.keys())]
# n_clusters=50
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_pos)
# node_clusters_coord=[[] for i in range(n_clusters)]
# node_clusters_i=[[] for i in range(n_clusters)]
# def get_cluster_center(center,node_coords,node_i,n_nodes=2):
#     return (sorted(zip(node_i,node_coords), key=lambda node_i : np.linalg.norm(node_i[1]-center)))[0:n_nodes]

# for i,(cluster, node) in enumerate(zip(kmeans.labels_,node_pos)):
#     node_clusters_coord[cluster].append(node)
#     node_clusters_i[cluster].append(i)
# test=cm.rainbow(np.arange(n_clusters))
# visualise_cluster(node_clusters_coord,test)

# #sample random 1 node from each cluster and train on each cluster left out
# #alternative is to take the center of the cluster
# center_cluster_inds = [list(zip(*get_cluster_center(center,node_coord,node_i)))[0]
#         for center,node_coord,node_i in zip(kmeans.cluster_centers_,node_clusters_coord,node_clusters_i)]
# #use one of both indices
# rand_cluster_ints = [np.random.choice(i_s,replace=False) for i_s in node_clusters_i]

# error_diff=[]
# for omit_sensor in range(len(rand_cluster_ints)):
#     X_test_omit=X.copy()
#     X_test_omit[:,omit_sensor]=np.zeros((len(X_test_omit)))

#     error_diff.append(test_GP(X_test_omit,Y))

# #normalise error diff
# values=[(error/max(error_diff))**8 for error in error_diff]
# visualise_cluster(node_clusters_coord,values=values)
# visualise_cluster(node_clusters_coord,values=values,view_angle=135)

# ##3

# #train on each cluster seperately and check accuracy
# #take the biggest amount of possible  ranodm points from cluster
# error_for_cluster=[]
# print("nr of training nodes, ", min([len(coords) for coords in node_clusters_coord]))
# rand_cluster_inds = [np.random.choice(i_s, size=min([len(coords) for coords in node_clusters_coord]),replace=False) for i_s in node_clusters_i]
# for i,inds in enumerate(rand_cluster_inds):
#     error_for_cluster.append(test_GP(X[:,inds],Y))

# values=[error/max(error_for_cluster) for error in error_for_cluster]
# visualise_cluster(node_clusters_coord,values=values)
# visualise_cluster(node_clusters_coord,values=values,view_angle=135)

# visualise_cluster(node_clusters_coord,values=error_for_cluster)

