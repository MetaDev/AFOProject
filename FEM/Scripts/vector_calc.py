#script contains many data structures from freecad, only to be imported from a freecad macre (FCMacro)

import numpy as np
import utility as ut


def getNodesFromFEMMeshSurface(femmesh, shape):
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
def getFaceFromNode(node_id, femmesh):
    for face in femmesh.Faces:
        if (node_id in femmesh.getElementNodes(face)):
            return face
#expects 3 vertexes in numpy format
#Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle    
def getNormalOfTriangle(vertexes):
    return normalize(np.cross(vertexes[1]-vertexes[0],vertexes[2]-vertexes[0]))
def getNormalOfNodes(node_ids,femmesh):
    #get all faces of a node
    nodeTriangles={id: [] for id in node_ids}
    triangleNormal = {}
    for face in femmesh.Faces:
        triangleNormal[face]=getNormalOfTriangle(getNPVertexFromFEMMeshface(femmesh, face))

        for id in node_ids:
            if (id in femmesh.getElementNodes(face)):
                nodeTriangles[id].append(face)
    
    #calculate average normal of each node
    return [np.average(np.array([triangleNormal[triangle_id] 
                        for triangle_id in nodeTriangles[node_id]]),axis=0) 
                        for node_id in node_ids]



#warning this method works only for strain vectors with smooth adjacent triangels
#strains are vectors according to the global coordinate system
#convert to local coordinate system of the normal of the respective triangle 
#project onto triangle plane
#rotate randomly (save the rotation as sensor configuration) around origin
def projectStrainVector3DOnMesh(strain_vec, tri_norm):
    #convert t`o local coordinate system by rotating (origin is the same, no translation)
    global_norm=np.array([0,0,1])
    R = getVectorToVectorRotation(global_norm,tri_norm)

    strain_vec = vectorToNPArr(strain_vec)
    
    #Ra=b, rotate 
    strain_vec_tri = np.dot(R,strain_vec)
    #project on triangle plane, our coordinate axis this plane
    proj_plane_strain=strain_vec_tri*np.array([1,1,0])
    #project on random axis of the triangle plane (x,y plane) do to coordinate system
    rand_theta = np.random.random()*2*np.pi
    rand_axis=np.array([np.cos(rand_theta),np.sin(rand_theta),0])
    
    #rand axis is a unit vector, thus the dot product isn't divided by it's length
    proj_axis_strain=rand_axis*(np.dot(proj_plane_strain,rand_axis))
    
    #return the size of this vector
    return np.linalg.norm(proj_axis_strain)

#not tested
#draw debug line from two freecad vectors
def debugLine(point1, point2):
    l=Part.Line(point1, point2)
    shape = l.toShape()
    Part.show(shape)
