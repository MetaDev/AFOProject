#script contains many data structures from freecad, only to be imported from a freecad macre (FCMacro)

import numpy as np
import utility as ut


def getNodesFromFEMMeshSurface(femmesh, shape):
    #nodes with a face
    
    return np.unique(ut.flatten( [(femmesh.getNodesByFace(f)) for f in shape.Faces ]))
def vectorToNPArr(v):
    return np.array([v.x, v.y, v.z])

#a and b are np arrays
def normalize(v):
    return v/np.linalg.norm(v)

def getVectorToVectorRotation(a, b):
    a = normalize(a)
    b = normalize(b)
  
    v = np.multiply(a, b)
    s = np.linalg.norm(v)
    #same vector
    if(s < 0.0001):
        return np.eye(3)
    c = np.dot(a, b)
    #opposite vector
    if(np.abs(c + 1) < 0.0001):
        R = np.eye(3)
        R[2, 2] = -1
        return R
    v_x = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    return np.eye(3) + v_x + (np.array(v_x)**2) * (1 / (1 + c))
#assumes that the first 3 nodes in a FEMMesh element are the definingvertexes of the triangle or face
#see: https://www.freecadweb.org/wiki/FEM_Mesh -> triangle element
def getNPVertexFromFEMMeshface(femmesh, face_id):
    six_node_ids = femmesh.getElementNodes(face_id)
    return np.array([vectorToNPArr(femmesh.Nodes[six_node_ids[0]]),
                    vectorToNPArr(femmesh.Nodes[six_node_ids[1]]),
                    vectorToNPArr(femmesh.Nodes[six_node_ids[2]])])
def getNodesFromFemMeshWithFace(femmesh):
    return np.unique([[id for id in femmesh.getElementNodes(f)] for f in femmesh.Faces])
#expects 3 vertexes in numpy format
#Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle    
def getNormalOfTriangle(vertexes):
    return normalize(np.cross(vertexes[1]-vertexes[0],vertexes[2]-vertexes[0]))
def getNormalOfFemmeshTriangles(femmesh):
    
    node_ids = getNodesFromFemMeshWithFace(femmesh)
    print(len(node_ids))
    #get all faces of a node
    nodeTriangles={id: [] for id in node_ids}
    triangleNormal = {}
    for face in femmesh.Faces:
        triangleNormal[face]=getNormalOfTriangle(getNPVertexFromFEMMeshface(femmesh, face))
        
        for id in femmesh.getElementNodes(face):
            nodeTriangles[id].append(face)
    
    #calculate average normal of each node
    return [np.average(np.array([triangleNormal[triangle_id] 
                        for triangle_id in nodeTriangles[node_id]]),axis=0) 
                        for node_id in node_ids]


#https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
def rand_triangle_coord_system(tri_n):
	#random coordinate system, with z axis as normal of triangle
	x = np.random.randn(3)  # take a random vector
	x -= x.dot(tri_n) * tri_n       # make it orthogonal to 
	x /= np.linalg.norm(x)  # normalize it
	y = np.cross(tri_n, x)
	return np.array([x,y,tri_n])
#https://stackoverflow.com/questions/21125987/how-do-i-convert-a-coordinate-in-one-3d-cartesian-coordinate-system-to-another-3
def project_3Dstrain_on_rand_1D_in_triangle(X,n_sensors):
	rand_CS = [rand_triangle_coord_system(normalize(np.random.rand(1,3))[0]) for i in range(n_sensors)]
	return np.array( [[np.dot(rand_CS,v) for i,v in enumerate(v_row)] for v_row in X ])

#warning this method works only for strain vectors with smooth adjacent triangels
#strains are vectors according to the global coordinate system
#given dict of strains per node and normal of node
#totest TODO
def projectStrainVector3DOnMesh(strain_vecs, coord_syst_node):
    return np.array( [np.dot(coord_syst_node[i],v) for i,v in enumerate(strain_vecs)] )

#not tested
#draw debug line from two freecad vectors
def debugLine(point1, point2):
    l=Part.Line(point1, point2)
    shape = l.toShape()
    Part.show(shape)
