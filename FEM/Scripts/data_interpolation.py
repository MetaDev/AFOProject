


disp_node_strain = np.array([ list(disp_dict.values()) for disp_dict in disp_node_i_strains ])

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

#old code for interpolation to simulate sensor position
 #the ranges to train the model on for small simple afo: x=0||2 , y=[0,5], z= [0,10]
#     import itertools
#     sens_pos=[[i,j,k] for (i,j,k) in itertools.product([0,2],np.linspace(0,5,3),np.linspace(0,10,4))]
# # if scipy_interp:
#             X = np.array([ func(sensor_positions) for func in strain_interp])
#         else:
#             #calculate interpolated strain value at each point
#             X = np.array([[calc_strain_on_point_in_mesh(pos,t__c_s) 
#                                         for pos in sensor_positions] 
#                                              for t__c_s in disp__triangles__coord_strain])