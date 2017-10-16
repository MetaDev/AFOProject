import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import scipy
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, normalize

import tensorflow as tf

from sklearn import linear_model

# read input
data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
result_path = r'C:\Users\Administrator\Google Drive\Windows\Research\Project\Docs\simple_AFO_results.xlsx'

n_sensor_axis = 1
axis_training = {1: [0], 2: [0, 1], 3: [0, 1, 2]}
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

#this data strucutre contains all the data strucutered in a mesh of triangles-> disp, mesh_triangles_by_coord_and_strain
#I only use the first 3 nodes of a triangle definition, these contain the corners, the other three are midpoints
#the 3 additional points are used for higher order interpolation
disp__triangles__coord_strain = np.array([
    
    [
        [   
            [nodes_i_coord[node_i] for node_i in triangle_nodes[0:3]]
        ,
            [nodes_strains[node_i] for node_i in triangle_nodes[0:3]]
        ]
        for triangle_nodes in mesh_triangles_nodes 
    ]
    
    for nodes_strains in disp_node_i_strains
]) 


#two options to interpolate 1, weighted sum of X closest points, interpolate function over all values with barycentric coordinates

def calc_uv(p,triangle):

    f1=triangle[0]-p
    f2=triangle[1]-p
    f3=triangle[2]-p
    #change triangles
    a=np.linalg.norm(np.cross(triangle[0]-triangle[1],triangle[0]-triangle[2]))
    a1=np.linalg.norm(np.cross(f2,f3))/a
    a2=np.linalg.norm(np.cross(f3,f1))/a
    a3=np.linalg.norm(np.cross(f1,f2))/a
    return [a1,a2,a3]

strain_interp=[ scipy.interpolate.LinearNDInterpolator(
            points=list(nodes_i_coord.values()),values=nodes_strain) for nodes_strain in disp_node_strain]

def calc_strain_on_point_in_mesh(point,triangles_coord_strain):
    #if any of the areas in calc_uv is outside of [0,1], the point is outside the triangle
    for triangle_c_s in triangles_coord_strain:
        uv=calc_uv(point,triangle_c_s[0])
        if all([0<=uv_<=1 for uv_ in uv]):
            return triangle_c_s[1][0]*uv[0] + triangle_c_s[1][1]*uv[1] + triangle_c_s[1][2]*uv[2]

    print("point not located on the mesh")
    #return some error if the point is not on the mesh

scalerX = StandardScaler()
scalerY = StandardScaler()
def preprocessing_data(X, Y, n_sensors=-1,sensor_positions=None, scipy_interp=True):
    # extract random n_sensor columns
    if n_sensors!=-1:
        X = np.delete(X, (tuple(np.random.choice(len(X[0]), len(X[0]) - n_sensors, replace=False))), axis=1)
    elif sensor_positions !=None:
        if scipy_interp:
            X = np.array([ func(sensor_positions) for func in strain_interp])
        else:
            #calculate interpolated strain value at each point
            X = np.array([[calc_strain_on_point_in_mesh(pos,t__c_s) 
                                        for pos in sensor_positions] 
                                             for t__c_s in disp__triangles__coord_strain])                                                                     
    # extract x axis value
    X = X[:, :, axis_training[n_sensor_axis]]
    # flatten X
    X = X.reshape(len(Y), -1)
    # scale to mean 0 and unit variance
    X = scalerX.fit_transform(X)
    noise=np.random.normal(0,0.1,np.shape(X))
    # X=X+noise
    Y = scalerY.fit_transform(Y)
    return X, Y


import itertools
sens_pos=[[i,j,k] for (i,j,k) in itertools.product([0,2],np.linspace(0,5,3),np.linspace(0,10,4))]

#pick a different layout for each trial for angle evaluation
X_pre, Y_pre = preprocessing_data(disp_node_strain,disp,10,sensor_positions=sens_pos)

import math

def autoencoder_reconstruct(dimensions,stop_gradient=False):
    with tf.variable_scope("ae"):
        #input to the network
        x_full = tf.placeholder(tf.float32, [None, dimensions[0]], name='x_f')
        x_partial = tf.placeholder(tf.float32, [None, dimensions[0]], name='x_p')
        current_input = x_partial

        # Build the encoder
        encoder = []
        for layer_i, n_output in enumerate(dimensions[1:]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(
                tf.random_uniform([n_input, n_output],
                                -1.0 / math.sqrt(n_input),
                                1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)
            current_input = output

        # latent representation
        z = current_input
        #if the ae and regessor are trained end to end stop the gradien at the 'middle'
        if stop_gradient:
            tf.stop_gradient(z)
        encoder.reverse()

        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
            W = tf.transpose(encoder[layer_i])
            b = tf.Variable(tf.zeros([n_output]))
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)
            current_input = output

        # now have the reconstruction through the network
        y = current_input

        # cost function is the me
        cost = tf.reduce_mean(tf.square(tf.subtract(y, x_full)))
    return {'x_p': x_partial,'x_f': x_full, 'z': z, 'y': y, 'cost': cost}

def regressor(dimensions,connected_input=None):
    with tf.variable_scope("regr"):
        if connected_input is not None:
            x=connected_input
        else:
            x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
        y = tf.placeholder(tf.float32, [None, dimensions[-1]], name='y')
        current_input = x
        for layer_i, n_output in enumerate(dimensions[1:]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(
                tf.random_uniform([n_input, n_output],
                                -1.0 / math.sqrt(n_input),
                                1.0 / math.sqrt(n_input)), name="w"+str(layer_i))
            b = tf.Variable(tf.zeros([n_output]), name="b"+str(layer_i))
            #the last layer should not have relu
            if layer_i==len(dimensions)-2:
                output = tf.matmul(current_input, W) + b
            else:
                output = tf.nn.relu(tf.matmul(current_input, W) + b)
            current_input = output
        # cost function is the me
        cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, output))))
    return {'x': x, 'y': y, 'cost': cost}

def train_ae_regr_end_to_end(X_partial,X,Y,n_epochs=500,verbose=False):
    ae = autoencoder_reconstruct(dimensions=[len(X[0]), 512,256, 32],stop_gradient=True)
    regr=regressor(dimensions=[len(X[0]),32,32,len(Y[0])],connected_input=ae['z'])
    ae_train_vars= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae')
    regr_train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regr')
    learning_rate=0.0001
    train_ae = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'], var_list= ae_train_vars)
    learning_rate=0.01
    train_regr = tf.train.AdamOptimizer(learning_rate).minimize(regr['cost'], var_list= regr_train_vars)
    both_opt = tf.group(train_ae, train_regr)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(n_epochs):
        sess.run(both_opt, feed_dict={ae['x_p']: X_partial,ae['x_f']: X,regr['y']: Y})
        #to get the values of the latent variables 
        # print(epoch_i, sess.run(ae['z'], feed_dict={ae['x']: X_pre}))
        if verbose:
            result=sess.run([ae['cost'],regr['cost']], feed_dict={ae['x_p']: X_partial,ae['x_f']: X,regr['y']: Y})
            print(epoch_i, "ae loss & regr loss", result[0],result[1])
            
    return ae,regr,sess


def test_sensor_reconstruction(X,Y,n_sensor,n_missing):
    #train on at least 2 data points
    if (n_sensor-n_missing<2):
        return
    print("n sensors and n missing", n_sensor,n_missing)
    #train and test 3 networks 1 on n_sens- missing sensors, two on the full amount of sensors but with one where there
    #are n_missing sensors ommitted and imputed by ae

    #cut and corrupt data according to parameters
    X=X[:,0:n_sensor]
    X_small=X[:,0:n_sensor-n_missing]
    #put all but a few values to 0
    default= np.zeros(((len(X),n_missing)))
    X_partial=np.copy(X)
    X_partial[:,(tuple(np.random.choice(len(X[0]), n_missing, replace=False)))] = default
    test_size=0.33
    #the training for the ae requires same indexing for both X sets
    trian_indices=np.random.choice(len(X),int(len(X)*(1-test_size)), replace=False)

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.33, random_state=42)
    X_small_train, X_small_test, Y_small_train, Y_small_test = train_test_split( X_small, Y, test_size=0.33, random_state=42)
    X_partial_train, X_partial_test,Y_partial_train, Y_partial_test = train_test_split( X_partial, Y, test_size=0.33, random_state=42)
    ae,regr,sess= train_ae_regr_end_to_end(X_partial_train,X_train,Y_train)

    #test
    result=sess.run([ae['cost'],regr['cost']], feed_dict={ae['x_p']: X_partial_test,ae['x_f']: X_test,regr['y']: Y_test})
    
    print("mse regr"+str(result[1]))
#evaluate
#get bounds of sensors positions

coords=np.array(list(nodes_i_coord.values()))
coord_bounds_x=(np.max(coords[:,0]),np.min(coords[:,0]))
coord_bounds_y=(np.max(coords[:,1]),np.min(coords[:,1]))
coord_bounds_z=(np.max(coords[:,2]),np.min(coords[:,2]))
import random
def random_sens_pos():
    random.seed(12345)
    return [random.uniform(*coord_bounds_x),random.uniform(*coord_bounds_y),random.uniform(*coord_bounds_z)]
rand_pos = [random_sens_pos() for i in range(100)]
import itertools
sens_pos=[[i,j,k] for (i,j,k) in itertools.product([0,2],np.linspace(0,5,8),np.linspace(0,10,8))]
#pick a different layout for each trial for angle evaluation
X_100, Y_100 = preprocessing_data(disp_node_strain,disp,sensor_positions=sens_pos)


test_sensor_reconstruction(X_100,Y_100,10,8)

#search different sensor amounts and sensors to be recovered
for n_sensor in range(5,50,6):
    for n_miss in range(1,50,6):
        test_sensor_reconstruction(X_100,Y_100,n_sensor,n_miss)