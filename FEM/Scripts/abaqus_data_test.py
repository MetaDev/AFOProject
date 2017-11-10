import numpy as np
import data_prep
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.model_selection as ms
import train_model
import data_relevance
# ANKLE_RM3 is het moment rond de enkelas, 
# ANKLE_UR3 is de verplaatsingshoek rond de enkelas, 
# UR zijn de angulaire verplaatsingsvectoren in alle vertices, 
# U zijn de verplaatsingsvectoren in alle vertices, 
# NE zijn de nominal strains in het AFO-oppervlak langs de lokale assen 
# (de 1-as is de Z-as geprojecteerd op het AFO-oppervlak, 
# de 3-as is de normaal op het AFO-oppervlak en 
# de 2-as is dan het kruisproduct van de 3-as en de 1-as)  
# gegeven als (NE_11, NE_22, NE_12) in alle integratiepunten (hier het midden van de quads), 
# MESH_CONNECTIVITY geeft voor elk element de vertexnummers en MESH_COORDINATES geeft voor elke vertex de coordinaten.
# Hier een voorbeeldje: df[‘NE’][200,10000,0] geeft NE_11 voor frame 200 voor het midden van het element 10000 en 
# df[‘ANKLE_UR3’][200] geeft de verplaatsingshoek rond de enkelas voor datzelfde frame
#In verband met de eenheden: lengtes zijn uitgedrukt in mm, hoeken in radialen, stresses in MPa en momenten in Nmm.
df = np.load("DF30.npz")

X=df["NE"]

mesh_conn = df["MESH_CONNECTIVITY"]

vertex_coord=df["MESH_COORDINATES"]
elts_coords=[np.mean([vertex_coord[elt] for elt in elts],axis=0)  for elts in mesh_conn]
Y=np.column_stack((df["ANKLE_UR3"],df["ANKLE_RM3"]))

n_sensors = 10
x_ind_rel= data_relevance.calc_most_significant_sensor(X,Y,test_method=1)
#TODO repeat this experiment for many sensors
for i in range(1,n_sensors):
    #the X is 3d , with the second axis being the number of sensors
    X_ind_rand = np.random.choice(len(X[0,:,0]), i, replace=False)
    X_reduced=X[:,np.random.choice(x_ind_rel[0:1000], i, replace=False),:]
    X_reduced_rand=X[:,X_ind_rand,:]
    X_reduced_first=X[:,list(range(i)),:]
    #flatten en normalise results
    X_reduced,_,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0])
    X_reduced_rand,_,_,_= data_prep.preprocessing_data(X_reduced_rand,Y, sensor_axis =[0])
    X_reduced_first,_,_,_= data_prep.preprocessing_data(X_reduced_first,Y, sensor_axis =[0])
    print(i)
    print("most relevant,",train_model.test_GP(X_reduced,Y))
    print("random sensors,",train_model.test_GP(X_reduced_rand,Y))
    print("first sensor,", train_model.test_GP(X_reduced_first,Y))

#visualise most relevant sensors
data_relevance.visualise_coords(np.array(elts_coords),colors=[0,0,0,1],size=1,view_angle=135)
data_relevance.visualise_coords(np.array(elts_coords)[x_ind_rel[0:1000]],colors=[0.8,0.6,0.5,1],size=100,view_angle=135)

#visualise clusters and most relevant clusters
n_clusters=50
cluster_node_inds,cluster_coords,cluster_centers=data_relevance.get_3D_clusters(elts_coords,n_clusters=n_clusters)

import matplotlib.cm as cm
test=cm.rainbow(np.arange(n_clusters))
data_relevance.visualise_cluster(cluster_coords,colors=test,size=1)

mses=[]
for i,inds in enumerate(cluster_node_inds):
    X_cluster=X[:,inds,:]
    #flatten en normalise results
    X_cluster,_,_,_= data_prep.preprocessing_data(X_cluster,Y, sensor_axis =[0])
    mses.append(train_model.test_GP(X_cluster,Y))
values=np.array(mses)
data_relevance.visualise_cluster(cluster_coords,values=values,size=1)
data_relevance.visualise_cluster(cluster_coords,values=values,size=1,view_angle=135)

