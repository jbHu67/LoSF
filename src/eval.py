from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from tqdm import tqdm
import os
import torch
import numpy as np
import numba
import torch.nn as nn
import trimesh
from scipy.spatial import cKDTree
from torch_geometric.nn import fps
from time import time

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import get_grid_coords
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from DCUDF.dcudf.mesh_extraction import Dcudf_on_UDF
from DCUDF.udf_models import Weighted_Dist_UDF

# set visible GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# os.environ["HYDRA_FULL_ERROR"] = "1"

log = RankedLogger(__name__, rank_zero_only=True)


def get_grid_coords(resolution: int = 128, length: float = 0.52):
    x = np.linspace(-length, length, resolution)
    y = np.linspace(-length, length, resolution)
    z = np.linspace(-length, length, resolution)
    grid = np.meshgrid(x, y, z)
    coords = np.stack(grid, axis=-1)
    coords = coords.reshape(-1, 3)
    return coords


def ball_query(pcd, queries, radius):
    # construct a KDTree
    tree = cKDTree(pcd)
    # indices = [tree.query_ball_point(query, r=radius) for query in queries]
    indices = tree.query_ball_point(queries, r=radius)
    return indices


def query_function(input_pcd, query_points, device):
    udf_model = Weighted_Dist_UDF()
    # udf_model = Weighted_BNN_UDF(K=40)
    device_id = device[-1]
    udf_model = nn.DataParallel(udf_model, device_ids=[int(device_id)])

    udf_model = udf_model.to(device)
    udf_model.load_state_dict(
        torch.load(
            os.path.join(
                "./DCUDF/pretrained_models/UDF/gf_sf_250", "udf_model_best.t7"
            ),
            map_location=torch.device(device),
        )
    )

    udf_model.eval()
    return udf_model.forward(input_pcd, query_points)


def local_patch_extract(
    pcd_dir, pcd_path, radius, has_noise, noise_level, has_outliers, outlier_ratio
):
    print("Extracting local patches...")
    pcd_name = pcd_path.split("/")[-1].split(".")[0]
    pcd = trimesh.load(pcd_path)
    verts = np.asarray(pcd.vertices)
    # # Downsample the point cloud
    # ratio = 0.5
    # sel_idx = fps(torch.tensor(verts), ratio=ratio)
    # verts = verts[sel_idx]
    # save downsampled point cloud
    pcd_save_dir = f"{pcd_dir}/../downsampled/"
    os.makedirs(pcd_save_dir, exist_ok=True)
    pcd_downsampled = trimesh.points.PointCloud(verts)
    pcd_downsampled.export(f"{pcd_save_dir}/{pcd_name}.ply")
    # normalize verst to [-0.5, 0.5]
    min_v = np.min(verts, axis=0)
    max_v = np.max(verts, axis=0)
    scale = 1 / np.max(np.abs(max_v - min_v))
    bias = (-0.5 * max_v - 0.5 * min_v) / (max_v - min_v)
    verts = verts * scale + bias
    pcd_change = None
    if has_noise:
        verts += np.random.normal(0, noise_level, verts.shape)
        # save noisy point cloud
        pcd_name = f"{pcd_name}_n{str(noise_level)}"
        pcd_change = trimesh.points.PointCloud(verts)
    if has_outliers:
        num_outliers = int(len(verts) * outlier_ratio)
        outlier_idx = np.random.choice(len(verts), num_outliers, replace=False)
        verts[outlier_idx] += np.random.normal(0, 0.1, (num_outliers, 3))
        # pull verts back to [-0.5, 0.5]
        verts[verts > 0.52] = 0.52
        verts[verts < -0.52] = -0.52
        pcd_name = f"{pcd_name}_o{str(outlier_ratio)}"
        pcd_change = trimesh.points.PointCloud(verts)
    if pcd_change is not None:
        pcd_save_dir = f"{pcd_dir}/../noisy_outlier/"
        os.makedirs(pcd_save_dir, exist_ok=True)
        pcd_change.export(f"{pcd_save_dir}/{pcd_name}.ply")
    coords = get_grid_coords(256)
    print("Creating kd-tree...")
    indices = ball_query(verts, coords, radius)
    print("ball query finished!")
    # filter index with less than 5 points
    pts_in_lengths = list(map(len, indices))
    pts_in_lengths = np.array(pts_in_lengths)
    filtered_query_idx = np.where(pts_in_lengths > 5)[0]
    filtered_indices = indices[filtered_query_idx]
    PatchVerts = np.zeros((len(filtered_indices), 128, 3))
    Queries = np.zeros((len(filtered_indices), 3))
    Queries_0 = np.zeros((len(filtered_indices), 3))
    ScaleFactors = np.zeros(len(filtered_indices))
    for i in tqdm(range(len(filtered_indices))):
        query_idx = filtered_query_idx[i]
        idx = filtered_indices[i]
        query_0 = coords[query_idx]
        verts_sel = verts[idx]
        # normalize to a sphere with radius 1
        verts_query = np.vstack([query_0, verts_sel])
        translation = np.mean(verts_query, axis=0)
        verts_query -= translation
        max_dist = np.max(np.linalg.norm(verts_query, axis=1))
        scale_factor = 1.0 / max_dist
        verts_query = verts_query * scale_factor
        query = verts_query[0]
        verts_sel = verts_query[1:]
        if len(idx) == 128:
            verts_sel = verts_sel
        elif len(idx) > 128:
            ratio = 128 / len(idx)
            sel_idx = fps(torch.tensor(verts_sel), ratio=ratio)
            # sel_idx = np.random.choice(len(idx), 128, replace=False)
            verts_sel = verts_sel[sel_idx]
        else:
            mean_v = np.mean(verts_sel, axis=0)
            padding_length = 128 - len(idx)
            verts_sel = np.vstack([verts_sel, np.tile(mean_v, (padding_length, 1))])
            # # upsample by fitting
            # x, y, z = verts_sel[:, 0], verts_sel[:, 1], verts_sel[:, 2]
            # A = 0.5 * np.vstack((x**2, y**2)).T
            # k_opt, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            # k1, k2 = k_opt
            # # randomly generate 256 points [x,y] in unit disk
            # r = np.sqrt(np.random.rand(256))
            # theta = np.random.rand(256) * 2 * np.pi
            # x = r * np.cos(theta)
            # y = r * np.sin(theta)
            # z = 0.5 * (k1 * x**2 + k2 * y**2)
            # ratio = 128 / 256
            # sel_idx = fps(torch.tensor(np.vstack([x, y, z]).T), ratio=ratio)
            # verts_sel = np.vstack([x, y, z]).T[sel_idx]
        PatchVerts[i] = verts_sel
        Queries[i] = query
        Queries_0[i] = query_0
        ScaleFactors[i] = scale_factor
    patch = {}
    patch["PatchVerts"] = PatchVerts.astype(np.float32)
    patch["Queries"] = Queries.astype(np.float32)
    # patch["Queries_0"] = Queries_0
    patch["ScaleFactors"] = ScaleFactors.astype(np.float32)
    patch["Queries_IDX"] = filtered_query_idx
    return patch


def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    if cfg.trainer.accelerator == "gpu":
        device_ids = trainer.device_ids
        device = f"cuda:{device_ids[0]}"
    elif cfg.trainer.accelerator == "cpu":
        device = "cpu"
    save_dir = cfg.paths.output_dir

    # parameters for DCUDF
    mesh_resolution = cfg.mesh.resolution
    threshold = cfg.mesh.threshold
    is_cut = cfg.mesh.is_cut
    laplacian_weight = cfg.mesh.laplacian_weight
    experiment_name = cfg.experiment.name
    save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    udf_save_dir = os.path.join(save_dir, "udf")
    mesh_save_dir = os.path.join(save_dir, "out_mesh")
    os.makedirs(udf_save_dir, exist_ok=True)
    os.makedirs(mesh_save_dir, exist_ok=True)
    checkpoint = torch.load(cfg.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    log.info("Starting testing!")

    pc_dir = cfg.data.data_dir
    radius = cfg.data.radius
    has_noise = cfg.data.has_noise
    noise_level = cfg.data.noise_level
    has_outliers = cfg.data.has_outliers
    outlier_ratio = cfg.data.outlier_ratio

    pc_path_list = []
    for file in os.listdir(pc_dir):
        if file.endswith(".ply"):
            pc_path_list.append(file)
    for i in range(len(pc_path_list)):
        print(f"Processing {i+1}/{len(pc_path_list)}>>>\n")
        pc_path = os.path.join(pc_dir, pc_path_list[i])
        data_name = pc_path.split("/")[-1].split(".pl")[0]
        # data_name = data_name.split("/")[-1]
        if os.path.exists(f"{udf_save_dir}/{data_name}_UDF.npy") is not True:
            start_time = time()
            patch_info = local_patch_extract(
                pc_dir,
                pc_path,
                radius,
                has_noise,
                noise_level,
                has_outliers,
                outlier_ratio,
            )
            end_time = time()
            patch_extract_time = end_time - start_time
            start_time = time()
            PatchVerts = patch_info["PatchVerts"]
            Queries = patch_info["Queries"]
            ScaleFactors = patch_info["ScaleFactors"]
            Queries_IDX = patch_info["Queries_IDX"]
            # transfer to tensor with GPU
            PatchVerts = torch.tensor(PatchVerts).float().to(device)
            Queries = torch.tensor(Queries).float().to(device)
            Queries = Queries.unsqueeze(1)
            Vecs_q = PatchVerts - Queries
            num_queries = Queries.shape[0]
            # split queries into batches
            batch_size = 2048 * 1
            num_batches = num_queries // batch_size
            if num_batches * batch_size < num_queries:
                num_batches += 1
            print("Predicting UDF...")
            PredUDF = np.zeros(num_queries)
            # DIS = np.zeros((batch_size * 100, 128))
            # j = 0
            for i in range(num_batches):
                start = i * batch_size
                end = np.min([start + batch_size, num_queries])
                input_data = {}
                input_data["verts"] = PatchVerts[start:end]
                input_data["vecs_q"] = Vecs_q[start:end]
                input_data["query"] = Queries[start:end]
                (
                    pred_udf,
                    displacement,
                ) = model(input_data, device)
                pred_udf = pred_udf.detach().cpu().numpy()
                PredUDF[start:end] = pred_udf[:, 0]
            #     displacement = displacement.detach().cpu().numpy()
            #     flag = np.random.randint(0, 100)
            #     if flag < 30 and j < 100:
            #         DIS[(j * batch_size) : ((j + 1) * batch_size)] = displacement
            #         j += 1
            # np.savez(f"{save_dir}/{data_name}_DIS.npz", DIS)
            grids = get_grid_coords(256)
            num_grids = grids.shape[0]
            UDF = np.zeros(num_grids) + 10
            PredUDF = PredUDF / ScaleFactors
            UDF[Queries_IDX] = PredUDF
            end_time = time()
            udf_infer_time = end_time - start_time
            print(
                f"{pc_dir}---{radius}---patch time: {patch_extract_time}, udf time: {udf_infer_time}"
            )

            save_data_udf = {"udf_vals": PredUDF, "queries_idx": Queries_IDX}
            np.save(f"{udf_save_dir}/{data_name}_UDF.npy", save_data_udf)
            UDF = torch.tensor(UDF).float().to(device)
        # np.save(f"{save_dir}/{data_name}_UDF.npy", UDF)
        else:
            udf_values = np.load(
                f"{udf_save_dir}/{data_name}_UDF.npy", allow_pickle=True
            ).item()
            grids = get_grid_coords(256)
            num_grids = grids.shape[0]
            UDF = np.zeros(num_grids) + 10
            Queries_IDX = udf_values["queries_idx"]
            PredUDF = udf_values["udf_vals"]
            UDF[Queries_IDX] = PredUDF
            UDF = torch.tensor(UDF).float().to(device)

        print("Extracting mesh...")
        extractor = Dcudf_on_UDF(
            query_function=query_function,
            udf_field=UDF,
            max_iter=300,
            resolution=mesh_resolution,
            threshold=threshold,
            is_cut=is_cut,
            bound_min=torch.tensor([-0.52, -0.52, -0.52]).to(device),
            bound_max=torch.tensor([0.52, 0.52, 0.52]).to(device),
            input_pcd=None,
            laplacian_weight=laplacian_weight,
            device=device,
        )
        mesh = extractor.optimize()
        # rotation_mat = trimesh.transformations.rotation_matrix(
        #     np.pi / 2, [0, 0, 1], mesh.centroid
        # )
        # mesh.apply_transform(rotation_mat)
        # mirror_mat = np.diag([-1, 1, 1, 1])
        # mesh.apply_transform(mirror_mat)
        mesh.export(f"{mesh_save_dir}/{data_name}_out.obj")
        torch.cuda.empty_cache()
        print(f"{data_name} finished!")


@hydra.main(
    version_base="1.3",
    config_path="../configs/experiment",
    config_name="eval.yaml",
)
def main(cfg: DictConfig) -> None:
    # extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
