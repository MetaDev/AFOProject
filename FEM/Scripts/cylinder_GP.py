import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, normalize


from sklearn import linear_model
import vector_calc as vc

# read input
data_path = "C:\\Users\\Administrator\\Google Drive\\Windows\\Research\\Project\\FEM\\Results\\data"
result_path = r'C:\Users\Administrator\Google Drive\Windows\Research\Project\Docs\simple_AFO_results.xlsx'
data_type = 2
n_sensor_axis = 1
axis_training = {1: [0], 2: [0, 1], 3: [0, 1, 2]}
# name_afo_project = "_cylinder"
# name_afo_project = "_rot_cube"
name_afo_project = "_simple_afo"
#the length (in dm) from the deformed object is used to calculate the angle based on displacement
length=0.1
if data_type == 0:
    x_file = "\\all_strain_list_simple" + name_afo_project
elif data_type == 1:
    x_file = "\\surface_strain_list" + name_afo_project
else:
    x_file = "\\proj_surface_strain_list" + name_afo_project
disp_vectors = "\\displacement_list"+ name_afo_project
#in mm
disp_coord = "\\node_coord"+ name_afo_project
faces_nodes = "\\faces_nodes"+ name_afo_project
X = np.load(data_path + x_file + ".npy")
Y = np.load(data_path + disp_vectors + ".npy")

nodes_i_coord = dict([tuple([tup4[0],tup4[1:4]]) for tup4 in np.load(data_path + disp_coord + ".npy")])

mesh_triangles_nodes = np.load(data_path + faces_nodes + ".npy")

#data structure with indexed node strains for each displacement
disp_node_i_strains=[[disp,dict([ (node_i, strain) for node_i, strain in zip(nodes_i_coord.keys(), strains)])]
                                for disp, strains in zip(Y,X)] 

#this data strucutre contains all the data strucutered in a mesh of triangles-> disp, mesh_triangles_by_coord_and_strain
#I only use the first 3 nodes of a triangle definition, these contain the corners, the other three are midpoints
#the 3 additional points are used for higher order interpolation
disp_triangles_coord_strain=[]
for disp_i_strains in disp_node_i_strains:
    disp=disp_i_strains[0]
    i_strains= disp_i_strains[1]
    triangles=[]
    for triangle_nodes in mesh_triangles_nodes:
        triangles.append(
            [[nodes_i_coord[node_i],i_strains[node_i]] for node_i in triangle_nodes[0:3]]
        )
    disp_triangles_coord_strain.append([disp,triangles])


disp__triangles__coord_strain = np.array([
    [disp_i_strains[0],
        [
            [   
                [nodes_i_coord[node_i] for node_i in triangle_nodes[0:3]]
            ,
                [disp_i_strains[1][node_i] for node_i in triangle_nodes[0:3]]
            ]
            for triangle_nodes in mesh_triangles_nodes 
        ]
    ]
    for disp_i_strains in disp_node_i_strains
]) 
print("test")
print(len(disp_triangles__coord_strain[0][1][0][0]))
#two options to interpolate 1, weighted sum of X closest points, interpolate function over all values with barycentric coordinates
# Y_func_1 =  lambda x, y : x + y
#2

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
def calc_strain_on_point_in_mesh(point,triangles_coord_strain):
    #if any of the areas in calc_uv is outside of [0,1], the point is outside the triangle
    for triangle_c_s in triangles_coord_strain:
        uv=calc_uv(point,triangle_c_s[0])
        if all([0<=uv_<=1 for uv_ in uv]):
            return [triangle_c_s[1][0]*uv[0],triangle_c_s[1][1]*uv[1],triangle_c_s[1][2]*uv[2]]


    #return some error if the point is not on the mesh
#the ranges to train the model on for small simple afo: x=0||2 , y=[0,5], z= [0,10]
calc_strain_on_point_in_mesh([0,2,2],disp__triangles__coord_strain[0][1])
#this function can be used to test interpolation as alternative to neural net
#encode a few points in the interpolation and estimate the rest
#a different function for each displacement
# Y_funcs=[ scipy.interpolate.LinearNDInterpolator(point s=Y_coord,values=X[i]) for i in range(len(X))]

print(Y_funcs[0](0,2,3))
# for i in range(3):
#     plt.hist(X[:,0,i], bins='auto')  # arguments are passed to np.histogram
#     plt.show()
scalerX = StandardScaler()
scalerY = StandardScaler()
def preprocessing_data(X, Y, n_sensors,sensor_positions=None):
    # extract random n_sensor columns
    if not sensor_position:
        X = np.delete(X, (tuple(np.random.choice(len(X[0]), len(X[0]) - n_sensors, replace=False))), axis=1)
    else:
        #calculate interpolated strain value at each point
        X = np.array([[calc_strain_on_point_in_mesh(pos,disp__t__c_s[1]) for pos in sensor_positions] for disp__t__c_s in disp__triangles__coord_strain])
    # extract x axis value
    X = X[:, :, axis_training[n_sensor_axis]]
    
    # flatten X
    X = X.reshape(len(Y), -1)
    
    # scale to mean 0 and unit variance
    X = scalerX.fit_transform(X)
    noise=np.random.normal(0,0.1,np.shape(X))
    X=X+noise
    Y = scalerY.fit_transform(Y)
    return X, Y
def evaluate_by_example(X_pre, Y,model, length):
    # #pick strain values from X, with a label from -40 to 20
    # to_pred_points = np.linspace(-4,2,7)

    # #find closest x axis displacement in label list
    # test_data_indexed = ([min(enumerate(Y), key=lambda x: abs(x[1][0]-pr)) for pr in to_pred_points])
    test_data_indexed=list(enumerate(Y))
    #predict with the strain values, calculate difference and convert to degrees
    pred_test_Y = model.predict(X_pre[np.array([id[0] for id in test_data_indexed])])
    #denormalise data
    pred_test_Y=scalerY.inverse_transform(pred_test_Y)
    test_Y=np.array([id[1] for id in test_data_indexed])
    
    #calculate the error given predicted data
    degree_error= [(calc_angle_degree(pr_y,length)-calc_angle_degree((t_y),length)) for pr_y,t_y in zip(pred_test_Y,test_Y)]
    #return list of avtual deformation angle in degrees and it's corresponding error
    return list(zip([calc_angle_degree((t_y),length) for t_y in test_Y],[e for e in degree_error]))

    # in my test setup the displacement is in dm and length in mm, thus I have to convert the length to dm by dividing by 100
def calc_angle_degree(displacement,length):
    #the first ordinal is x, the second y
    return np.rad2deg(np.tanh(displacement/length))

# Instanciate a Gaussian Process model
kernel = 1.0 * RBF() \
    + WhiteKernel()

gp = GaussianProcessRegressor(kernel=kernel,alpha=0, n_restarts_optimizer=9)
# # Initialize model.
# logreg = linear_model.LinearRegression()
# X_pre, Y_pre = preprocessing_data(X, Y,10)
# # # Use cross_val_score to automatically split, fit, and score.
# scores = ms.cross_val_score(gp, X_pre, Y_pre, cv=10,scoring="r2")
# print(scores)
n_trials = 1
cv_results = []
degree_test= []
for i, n_s in enumerate(range(10, 11)):
    # for evaluation of a model accuracy the sensor layout should be kept constant over trials 
    # X_pre, Y_pre = preprocessing_data(X, Y, n_s)

    mse=[]
    for n in range(n_trials):
        #pick a different layout for each trial for angle evaluation
        X_pre, Y_pre = preprocessing_data(X,Y,n_s)
        # to average out differences in results of the model, NN trains stochastically
        model = gp
        # fix random seed for reproducibility
        seed = 7
        kfold = ms.KFold(n_splits=5, random_state=seed)
        cv_score_mse = []

        for train, test in kfold.split(X_pre, Y_pre):
            # Fit the model
            model.fit(X_pre[train], Y_pre[train])
            # evaluate the model

            y_mean, y_cov = gp.predict(X_pre[test], return_cov=True)
            y_error = np.mean((y_mean-Y_pre[test])**2)
            mse.append(y_error * 100)
            
        # degree_test.append(evaluate_by_example(X_pre, Y,model))
        cv_score_mse.append([np.mean(mse),np.std(mse)])
        
    #calc mean and std from degree test
    degree_test.append(evaluate_by_example(X_pre, Y,model,length))
    cv_results.append([n_s, np.mean(np.array(cv_score_mse)[:,0]),
    np.mean(np.array(cv_score_mse)[:,1])])
print(cv_results)
#plot heatmap
test_mean=np.mean(degree_test,axis=0).reshape(len(degree_test[0]),-1)

A =np.stack((
     test_mean[:,0],
     test_mean[:,1],
     test_mean[:,2]+test_mean[:,3]),axis=1)
from scipy.interpolate import griddata
# grid the data.
# define grid.
xi = np.linspace(-60,60,100)
yi = np.linspace(-60,60,100)
zi = griddata(( test_mean[:,0],  test_mean[:,1]), test_mean[:,2]+test_mean[:,3],(xi[None,:], yi[:,None]), method='cubic')
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
#plt.savefig(r"C:\Users\Administrator\Google Drive\Windows\Research\Project\FEM\Results\learning\visual\heatmap_error_rectangular.png")
plt.show()
