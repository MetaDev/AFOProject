from sklearn.preprocessing import StandardScaler, normalize
import numpy as np

import pickle

sensor_axis={1: [0], 2: [0, 1], 3: [0, 1, 2]}
def read_data():
    # read input
    data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
    result_path = r'C:\Users\Administrator\Google Drive\Windows\Research\Project\Docs\simple_AFO_results.xlsx'

    # name_afo_project = "_cylinder"
    # name_afo_project = "_rot_cube"
    name_afo_project = "_simple_afo"
    #the length (in dm) from the deformed object is used to calculate the angle based on displacement
    length=0.1

    x_file = "\\all_strain_list" + name_afo_project

    disp_vectors = "\\displacement_list"+ name_afo_project
    #in mm
    node_coord = "\\nodes_coord"+ name_afo_project
    faces_nodes = "\\faces_as_nodes"+ name_afo_project

    with open(data_path + x_file+ ".pickle", 'rb') as handle:
        #dict, the encoding is because I am writing with pickle in python 2 and reading in python 3
        disp_node_i_strains = pickle.load(handle, encoding='iso-8859-1')

    with open(data_path + disp_vectors+ ".pickle", 'rb') as handle:
        #list
        Y = pickle.load(handle, encoding='iso-8859-1')
    with open(data_path + node_coord+ ".pickle", 'rb') as handle:
        #dict, the index of the dict is casted to a float, thus has to be cast back
        nodes_i_coord = { int(key) : value for key, value in pickle.load(handle, encoding='iso-8859-1').items()}
    with open(data_path + faces_nodes+ ".pickle", 'rb') as handle:
        #list
        mesh_triangles_nodes = pickle.load(handle, encoding='iso-8859-1')

    disp_node_strain = np.array([ list(disp_dict.values()) for disp_dict in disp_node_i_strains ])
    disp = np.array(Y)
    return disp_node_strain,disp,nodes_i_coord
#TODO change name to normalisation and flatten, remove sensor extraction functionality
def preprocessing_data(X, Y, n_sensors=-1, sensor_axis =[0, 1, 2]):
    
    
    scalerX = StandardScaler()
    scalerY = StandardScaler()
    # extract random n_sensor columns, -1 means no set reduction
    if n_sensors!=-1:
        X =X[:,np.random.choice(len(X[0]), n_sensors, replace=False)]
                                                                    
    # extract x axis value
    X = X[:, :, sensor_axis]
    # flatten X
    X = X.reshape(len(Y), -1)
    # scale to mean 0 and unit variance
    X = scalerX.fit_transform(X)
    noise=np.random.normal(0,0.1,np.shape(X))
    # X=X+noise
    Y = scalerY.fit_transform(Y)
    return X, Y,scalerX, scalerY
