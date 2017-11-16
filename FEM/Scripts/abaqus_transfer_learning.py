import numpy as np
import data_prep
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.model_selection as ms
import train_model
import data_relevance

df = np.load("DF30.npz")
#import PF30 as well and combine data sets

X=df["NE"]
mesh_conn = df["MESH_CONNECTIVITY"]

vertex_coord=df["MESH_COORDINATES"] #in mm
elts_coords=[np.mean([vertex_coord[elt] for elt in elts],axis=0)  for elts in mesh_conn]

Y=np.reshape(df["ANKLE_UR3"],(-1,1))

n_sensors = 10
##1S
##calculate accurcay difference between choosing relevant random and first sensors
x_ind_rel= data_relevance.calc_most_significant_sensor(X,Y,test_method=1)

#use different layouts, different combinations of clusters (points from relevant clusters)

n_clusters=50
cluster_node_inds,cluster_coords,cluster_centers=data_relevance.get_3D_clusters(elts_coords,n_clusters=n_clusters)
#TODO filter only relevant custers
mses=[]
for i,inds in enumerate(cluster_node_inds):
    X_cluster=X[:,inds,:]
    #flatten en normalise results
    X_cluster,Y_norm,_,_= data_prep.preprocessing_data(X_cluster,Y, sensor_axis =[0])
    mses.append(train_model.test_GP(X_cluster,Y_norm))
#
n_sensors_per_cluster=2
sensor_layout=np.array([np.random.choice(inds,n_sensors_per_cluster) for inds in cluster_node_inds]).flatten()



#add noise to the position of 
elts_coords=np.array(elts_coords)
noise = np.random.normal(0, np.min(np.var(elts_coords,axis=None))/10, elts_coords.shape)
elts_coords_noise = elts_coords + noise
#do noisy interpolation and train on that data
strain_interp=[ scipy.interpolate.LinearNDInterpolator(
            points=list(nodes_i_coord.values()),values=nodes_strain) for nodes_strain in disp_node_strain]
