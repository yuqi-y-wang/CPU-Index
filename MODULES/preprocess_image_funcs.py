import numpy as np
import pywavefront
import nibabel as nib

from . import preprocess_dict

'''
This function turns obj (LPS) coordinates into voxel coordinates
Note that the NIFTI uses RAS, will transfer later
INPUT:
    xyz: a 1x3 structure of obj coordinates
    nifti_img_affine: the affine matrix of the corresponding nifti
    
'''
def obj_coord2_nifti(xyz, nifti_img_affine, point=True):
    M = nifti_img_affine[:3, :3]
    b = nifti_img_affine[:3, 3]
    if point:
        xyz = np.array(xyz)
        ## turn wavefront obj LPS to NIFTI RAS 
        xyz[0] *= -1
        xyz[1] *= -1
        xyz = np.array(xyz)-b
    else:
        xyz = np.array(xyz)
    return tuple(np.linalg.solve(M, xyz))

'''
This function takes into the 
INPUT: 
    nifti_path and annotation path
OUTPUT:

'''
def process_obj(nifti_path, annotation_path):
    nifti_img = nib.load(nifti_path)
    scene = pywavefront.Wavefront(annotation_path)
    vertices =[obj_coord2_nifti(xyz, nifti_img.affine) for xyz in scene.vertices]
    img_arr = nifti_img.get_fdata()
    n_lesions = int(len(scene.vertices)/8)
    mask = np.zeros_like(img_arr)
    sizes = []
    location = np.zeros(preprocess_dict.num_parts*3,)
    for i in range(0,n_lesions):
        x_points = []
        y_points = []
        z_points = []

        for j in range(0,8):
            x_points.append(vertices[i*8+j][0])
            y_points.append(vertices[i*8+j][1])
            z_points.append(vertices[i*8+j][2])

        x_range = max(x_points)-min(x_points)
        y_range = max(y_points)-min(y_points)
        z_range = max(z_points)-min(z_points)
        sizes.append(x_range*y_range*z_range)
        
        x_min = np.floor(min(x_points)).astype(int)
        x_max = np.ceil(max(x_points)).astype(int)+1
        y_min = np.floor(min(y_points)).astype(int)
        y_max = np.ceil(max(y_points)).astype(int)+1
        z_min = np.floor(min(z_points)).astype(int)
        z_max = np.ceil(max(z_points)).astype(int)+1
        
        x_center = (x_min+x_max)/2
        y_center = (y_min+y_max)/2
        z_center = (z_min+z_max)/2
        # print(np.array([x_center/img_arr.shape[0],y_center/img_arr.shape[1],z_center/img_arr.shape[2]]))
        pos_1 = np.ceil(preprocess_dict.num_parts*np.array([x_center/img_arr.shape[0],y_center/img_arr.shape[1],z_center/img_arr.shape[2]])).astype(int)
        for j in range(len(pos_1)):
            location[pos_1[j]+preprocess_dict.num_parts*j-1] += 1

    #     print(x_min,x_max, y_min,y_max, z_min,z_max)
        mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1
    return n_lesions, sizes, list(location), mask