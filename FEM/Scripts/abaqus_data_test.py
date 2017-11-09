import numpy as np
import data_prep
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.model_selection as ms
import train_model
import data_relevance
df = np.load("DF30.npz")
print(df.files)
print(df["NE"][200,10000])
print(np.shape(df["NE"]))
X=df["NE"]
Y=np.column_stack((df["ANKLE_UR3"],df["ANKLE_RM3"]))
print(np.shape(Y))
n_sensors = 10
for i in range(1,n_sensors):
    #the X is 3d , with the second axis being the number of sensors
    X_ind_rand = np.random.choice(len(X[0,:,0]), i, replace=False)
    # data_relevance.calc_most_significant_sensor(X,Y,n_sensors=i)
    print(i)
    #TODO find out why the first sensors contain a lot of data and the others not
    X_reduced=X[:,list(range(i)),:]
    X_reduced,_,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0, 1, 2])
    print(train_model.test_GP(X_reduced,Y))
# X_reduced=X[:,list(range(10)),:]
# X_reduced,Y,_,_= data_prep.preprocessing_data(X_reduced,Y, sensor_axis =[0, 1, 2])
# print(train_model.test_GP(X_reduced,Y))
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