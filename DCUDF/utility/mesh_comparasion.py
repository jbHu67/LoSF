"""
Compare the ground truth Meshes with the Deep Learning model generated meshes in different ways
    
    1. Chamfer Distance
    2. 


"""


import os
import argparse
import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance

from mesh_to_pcd import mesh_to_pcd
from chamferDistance import get_chamfer_dist



def chamfer_distance_ply(o3d_pcd_1, o3d_pcd_2):
    points_1 = np.asarray(o3d_pcd_1.vertices)
    points_2 = np.asarray(o3d_pcd_2.vertices)
    num_of_pcd_1 = points_1.shape[0]
    num_of_pcd_2 = points_2.shape[0]
    pcd_tensor_1 = torch.zeros((1, num_of_pcd_1, 3))
    pcd_tensor_2 = torch.zeros((1, num_of_pcd_2, 3))

    pcd_tensor_1[0] = torch.from_numpy(points_1)
    pcd_tensor_1[0] = torch.from_numpy(points_2)

    return chamfer_distance(pcd_tensor_1, pcd_tensor_2)[0]

def calculate_chamfer_distance(gt_mesh_path, test_mesh_path):
    gt_pcd = mesh_to_pcd(gt_mesh_path, save_pcd=False, num_of_points=48000)
    test_pcd = mesh_to_pcd(test_mesh_path, save_pcd=False, num_of_points=48000)


    return chamfer_distance_ply(gt_pcd, test_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_mesh_path', required=True, help='The Path to the ground truth mesh')
    parser.add_argument('--test_mesh_path', required=True, help='The Path to the model generated mesh')

    arg = parser.parse_args()

    gt_mesh_path = arg.gt_mesh_path
    test_mesh_path = arg.test_mesh_path

    print(get_chamfer_dist(arg.gt_mesh_path, arg.test_mesh_path))

