import numpy as np
import data_prep

import sklearn.model_selection as ms
import train_model
import data_relevance

df = np.load("ankle/DF30.npz")
#import PF30 as well and combine data sets

X=df["NE"]
mesh_conn = df["MESH_CONNECTIVITY"]

vertex_coord=df["MESH_COORDINATES"] #in mm
elts_coords=[np.mean([vertex_coord[elt] for elt in elts],axis=0)  for elts in mesh_conn]

Y=np.reshape(df["ANKLE_UR3"],(-1,1))

n_sensors = 10

x_ind_rel= data_relevance.calc_most_significant_sensor(X,Y,test_method=1)
##1
#use different layouts, different combinations of clusters (points from relevant clusters)

n_clusters=50
cluster_node_inds,cluster_coords,cluster_centers=data_relevance.get_3D_clusters(elts_coords,n_clusters=n_clusters)
#get most relevant clusters
n_best_clusters=10
best_clusters=set()
cluster_ind_node_inds=list(enumerate(cluster_node_inds))
ind=0
while len(best_clusters)<n_best_clusters:
    c_i=[c_i_n_is[0] for c_i_n_is in cluster_ind_node_inds if x_ind_rel[ind] in c_i_n_is[1] ][0]
    best_clusters.add(c_i)
    ind+=1
best_cluster_node_inds=np.array(cluster_node_inds)[list(best_clusters)]


#1.1
#This simulates a case of noisy sensors where some sense of sensor position (it's cluster) is preserved
# make random configurations, make a train and test set of these configurations
# with enough training data (2000 000) the mse reaches reasonable (<0.1) values
n_sensors_per_cluster=2
max_n_layouts=4000
X_layouts=[]
Y_layouts=[]
for n_layouts in range(100,max_n_layouts,200):
    for i in range(n_layouts):
        sensor_layout_inds=np.array([np.random.choice(inds,n_sensors_per_cluster) for inds in best_cluster_node_inds]).flatten()
        #use indices to create sets
        X_reduced=X[:,sensor_layout_inds,:]
        X_reduced,Y_norm,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0])
        X_layouts.extend(X_reduced)
        Y_layouts.extend(Y_norm)
    #GP has higher error than NN
    #print(train_model.test_GP(X_layouts,Y_layouts)[0])
    print("number of samples", len(X_layouts))
    history,model=train_model.test_nn_regr(np.array(X_layouts),np.array(Y_layouts),layers=[32,32,32])
    print("mse", history.history['val_mean_squared_error'][-1])


##1.2
##TODO run on gpu nodes with a lot of data, learning seems to increase
#more advanced transfer learning by permuting the layouts
#takes way more data to train (runs for 100 seconds to reach mse 0.8)
# n_sensors_per_cluster=2

# X_layouts=[]
# Y_layouts=[]
# from random import shuffle
# n_layouts=4000

# for i in range(n_layouts):
#     sensor_layout_inds=np.random.permutation(np.array([np.random.choice(inds,n_sensors_per_cluster) 
#                 for inds in best_cluster_node_inds]).flatten())
#     #use indices to create sets
#     X_reduced=X[:,sensor_layout_inds,:]

#     X_reduced,Y_norm,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0])
#     X_layouts.extend(X_reduced)
#     Y_layouts.extend(Y_norm)

# history,model=train_model.test_nn_regr(np.array(X_layouts),np.array(Y_layouts),layers=[32,64,32])
# print(history.history['val_mean_squared_error'][-1])


##1.3
##tested mean and variance of latyour data as features to train, but it doesn't train