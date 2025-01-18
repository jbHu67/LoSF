import time
import numba
import open3d as o3d
import argparse
import trimesh
from udf_models import *
from dcudf.mesh_extraction import Dcudf_on_UDF
from utility.mesh_to_pcd import mesh_to_pcd
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch3d.ops
import pytorch3d
import numpy as np
import sys
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_udf(udf_model, input_pcd, query):
    """
    udf_model: trained UDF model
    input_pcd: input point cloud with shape: (1, Num_points, 3)
    query: query points with shape: ()
    """
    query = query.unsqueeze(0).transpose(1, 2)  # (1,C,M)
    pred_udf = udf_model(input_pcd, query)

    return pred_udf.squeeze(0)


def get_knn_dist(dense_pcd, query):
    # dense_pcd: N,3
    # query:M,3

    dense_pcd = dense_pcd.unsqueeze(0)  # (1,N,3)
    query = query.unsqueeze(0)  # (1,M,3)
    dists, _, _ = pytorch3d.ops.knn_points(
        query, dense_pcd, K=1, return_nn=True, return_sorted=False
    )  # (1,M,1)
    dists = dists.squeeze(2).squeeze(0)  # (M)
    return torch.sqrt(dists)


def get_udf_grids(N, input_pcd, input_udf):
    max_batch = 2**16

    size = 1.05
    voxel_size = size / (N - 1)

    grids_verts = np.mgrid[:N, :N, :N]
    grids_verts = np.moveaxis(grids_verts, 0, -1)

    grids_coords = grids_verts / (N - 1) * size - size / 2

    grids_coords_flatten = np.asarray(
        grids_coords.reshape(-1, 3), dtype=np.float64
    )  # (N**3, 3)

    grids_udf_flatten = np.zeros((N**3,))

    num_samples = N**3

    head = 0

    while head < num_samples:
        sample_subset = (
            torch.from_numpy(
                grids_coords_flatten[head : min(head + max_batch, num_samples), :]
            )
            .cuda()
            .float()
        )
        with torch.no_grad():
            df = get_knn_dist(input_pcd.squeeze(0).reshape(-1, 3), sample_subset)

        grids_udf_flatten[head : min(head + max_batch, num_samples)] = (
            df.detach().cpu().numpy()
        )
        head = head + max_batch

    norm_mask = grids_udf_flatten < voxel_size * 2
    norm_idx = np.where(norm_mask)[0]
    head, num_samples = 0, norm_idx.shape[0]

    while head < num_samples:
        sample_subset_mask = np.zeros_like(norm_mask)
        sample_subset_mask[norm_idx[head : min(head + max_batch, num_samples)]] = True
        sample_subset = (
            torch.from_numpy(grids_coords_flatten[sample_subset_mask, :]).cuda().float()
        )

        with torch.no_grad():

            df = get_udf(input_udf, input_pcd=input_pcd, query=sample_subset)

        grids_udf_flatten[sample_subset_mask] = (
            df.reshape((df.shape[0])).detach().cpu().numpy()
        )

        head = head + max_batch

    grids_udf = grids_udf_flatten.reshape(N, N, N)

    return voxel_size, grids_coords, grids_udf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="input point cloud path, ply file support",
    )

    parser.add_argument(
        "--mesh_path",
        default="./experiments_on_mesh/342.obj",
        help="the path to the ground truth mesh file to generate the dense pcd",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="output triangle mesh path",
    )

    parser.add_argument("--res", type=int, default=128)

    parser.add_argument(
        "--scale",
        type=bool,
        default=False,
        help="whether scale the input into a unit cube",
    )

    parser.add_argument(
        "--resolution", type=int, default=256, help="the resolution for the output mesh"
    )

    parser.add_argument(
        "--sample_points",
        type=int,
        default=48000,
        help="the number of points to sample from mesh",
    )


    parser.add_argument(
        "--udf_input",
        type=str,
        default=None,
        help="the path to the other models generated UDF",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default='./pretrained_models/UDF/gf_sf_250',
        help="where you put the trained model",
    )

    parser.add_argument(
        "--save_UDF", type=bool, default=False, help="save the predicted UDF as np file"
    )

    parser.add_argument(
        "--add_noise",
        type=bool,
        default=False,
        help="whether add noise to the input point cloud",
    )

    arg = parser.parse_args()

    if arg.input is None:
        # This means the user input is a ground truth mesh file instead of
        # point cloud file
        sampled_pcd = mesh_to_pcd(
            arg.mesh_path, num_of_points=arg.sample_points, save_pcd=False
        )
        sampled_pcd_np = np.asarray(sampled_pcd.vertices)  # [48000, 3]
        # test_pcd_data = np.load("E:/training_surfaces/smooth/17.npz")
        # sampled_pcd_np = test_pcd_data['Surf_pts']
    
    else:
        pcd_o3d = o3d.io.read_point_cloud(arg.input)
        sampled_pcd_np = np.asarray(pcd_o3d.points)

    if arg.scale:

        pcd_max = np.max(sampled_pcd_np, axis=0, keepdims=True)
        pcd_min = np.min(sampled_pcd_np, axis=0, keepdims=True)

        center = (pcd_max + pcd_min) / 2
        scale = np.max(pcd_max - pcd_min)

        sampled_pcd_np = (sampled_pcd_np - center) / scale

    dense_pcd = torch.from_numpy(sampled_pcd_np).unsqueeze(0).cuda().float()

    if arg.add_noise:
        dense_pcd += 0.005 * torch.randn_like(dense_pcd)
        dense_pcd.clamp_(min=-1, max=1)

    
    use_knn = True
    if use_knn == True:
        udf_model = Weighted_Dist_UDF()
        # udf_model = Weighted_BNN_UDF(K=40)
        udf_model = nn.DataParallel(udf_model)

        udf_model = udf_model.cuda()
        udf_model.load_state_dict(
            torch.load(os.path.join(arg.model_path, "udf_model_best.t7"))
        )

        udf_model.eval()

        def query_function(input_pcd, query_points, model='KNN'):

            if model == 'BNN':
                udf, valid_index, valid_K_count = udf_model.forward(input_pcd, query_points)
                filerted_udf = torch.full(udf.size(), fill_value=50.0, device='cuda')
                if valid_index.numel()==0:
                    return filerted_udf, 0, 0
                else:
                    filerted_udf[valid_index[:,0], valid_index[:,1]] = udf[valid_index[:,0], valid_index[:,1]]
                    return filerted_udf, valid_index, valid_K_count
            
            elif model == 'KNN':
                return udf_model.forward(input_pcd, query_points)

    
    resolution = arg.resolution
    threshold = 0.003

    # we have a lot default parameters, see source code for details.
    pointcloud = dense_pcd.reshape((dense_pcd.shape[1], 3)).cpu().numpy()
    object_bbox_min = np.array(
        [np.min(pointcloud[:, 0]), np.min(pointcloud[:, 1]), np.min(pointcloud[:, 2])])-0.05
    object_bbox_max = np.array(
        [np.max(pointcloud[:, 0]), np.max(pointcloud[:, 1]), np.max(pointcloud[:, 2])])+0.05
    
    extractor = Dcudf_on_UDF(
        query_function=query_function,
        udf_field_path=arg.udf_input,
        max_iter=10,
        resolution=resolution,
        threshold=threshold,
        is_cut=False,
        # bound_min=torch.tensor([ 0.52,0.52, 0.52]),
        # bound_max=torch.tensor([ 0.52,0.52, 0.52]),
        bound_min=torch.from_numpy(object_bbox_min),
        bound_max=torch.from_numpy(object_bbox_max),
        input_pcd=dense_pcd.reshape((dense_pcd.shape[1], 3)),
        laplacian_weight=4000
    )
    mesh = extractor.optimize()

    mesh.export(arg.output)
    torch.cuda.empty_cache()
