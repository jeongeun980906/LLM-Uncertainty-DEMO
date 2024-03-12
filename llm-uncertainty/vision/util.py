import numpy as np

def get_det_points(depth_pixel, label_pixel, transform_mat):
    detection_num=len(np.unique(label_pixel))
    print(np.unique(label_pixel))
    # Total cluster loop
    centers = []
    for cluster_idx in np.unique(label_pixel)[1:]:
        # Random color 
        # color = random_color_gen()
        cluster_xlst, cluster_ylst = np.where(label_pixel==cluster_idx)
        for idx, (x,y,z) in enumerate(zip(depth_pixel[cluster_xlst, cluster_ylst,2], \
                                            depth_pixel[cluster_xlst, cluster_ylst,0], \
                                            depth_pixel[cluster_xlst, cluster_ylst,1])):

            position=camera_to_base(transform_mat, np.array([[x,y,z]])).reshape(-1)
            if idx==0: single_cluster=np.array([[position[0],position[1],position[2]]])
            else:
                if position[2]<0.72 or position[0]>1.3 or position[1]>0.45 or position[1]<-0.45: # Remove unnecessary part
                    continue 
                single_cluster = np.concatenate((single_cluster,np.array([[position[0],position[1],position[2]]])))
        
        # if cluster_idx==0: total_clusters=single_cluster 
        # else: total_clusters = np.concatenate((total_clusters, single_cluster))
        cen_x = np.average(single_cluster[:,0])-0.1; cen_y=np.average(single_cluster[:,1]); 
        cen_z= 0.82 # np.average(single_cluster[:,2])
        center = [cen_x,cen_y,cen_z]
        centers.append(center)
        print("Object Center", center)
    return centers

def camera_to_base(transform_mat, points):
    ones = np.ones((len(points),1))
    points = np.concatenate((points,ones),axis=1)
    t_points = points.T
    t_transformed_ponints = np.dot(transform_mat,t_points)
    transformed_ponints = t_transformed_ponints.T
    xyz = transformed_ponints[:,0:3]
    return xyz



def Rotation_X(rad):
    roll = np.array([[1, 	       0, 	      0,    0],
             		 [0, np.cos(rad), -np.sin(rad), 0],
             		 [0, np.sin(rad),  np.cos(rad), 0],
             		 [0,		   0,	      0,    0]])
    return roll 


def Rotation_Y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad), 0],
              		  [0,		    1, 	         0, 0],
              		  [-np.sin(rad),0, np.cos(rad), 0],
              		  [0, 		    0, 	         0, 0]])
    return pitch


def Rotation_Z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0, 0],
         	        [np.sin(rad),  np.cos(rad),  0, 0],
              		[0, 			         0,  1, 0],
             		[0, 			         0,  0, 0]])
    return yaw 

def Translation(x , y, z):
    Position = np.array([[0, 0, 0, x],
                         [0, 0, 0, y],
                         [0, 0, 0, z],
                         [0, 0, 0, 1]])
    return Position


def HT_matrix(Rotation, Position):
    Homogeneous_Transform = Rotation + Position
    return Homogeneous_Transform