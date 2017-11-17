import numpy as np
import data_prep
from sklearn.gaussian_process import GaussianProcessRegressor
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
n_sensors_per_cluster=2
n_layouts=400
#make random configurations, make a train and test set of these configurations
X_layouts=[]
Y_layouts=[]
for i in range(n_layouts):
    sensor_layout_inds=np.array([np.random.choice(inds,n_sensors_per_cluster) for inds in best_cluster_node_inds]).flatten()
    #use indices to create sets
    X_reduced=X[:,sensor_layout_inds,:]
    X_reduced,Y_norm,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0])
    X_layouts.extend(X_reduced)
    Y_layouts.extend(Y_norm)
print(len(Y_layouts))
#GP has higher error than NN
#print(train_model.test_GP(X_layouts,Y_layouts)[0])
history,model=train_model.test_nn_regr(np.array(X_layouts),np.array(Y_layouts))
print(history.history['val_acc'][-1])
#test with unseen layout
sensor_layout_inds=np.array([np.random.choice(inds,n_sensors_per_cluster) for inds in best_cluster_node_inds]).flatten()
X_reduced=X[:,sensor_layout_inds,:]
X_reduced,Y_norm,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0])
scores = model.evaluate(X_reduced, Y_norm)
print(model.metrics_names[1], scores[1])
#2 now try to improve training with interpolation
#do noisy interpolation and train on that data
#add noise to the position of the nodes
# elts_coords=np.array(elts_coords)
# noise = np.random.normal(0, np.min(np.var(elts_coords,axis=None))/10, elts_coords.shape)
# elts_coords_noise = elts_coords + noise
# #Define which points will be used for interpolation
# strain_interp=[ scipy.interpolate.LinearNDInterpolator(
#             points=list(nodes_i_coord.values()),values=nodes_strain) for nodes_strain in disp_node_strain]
