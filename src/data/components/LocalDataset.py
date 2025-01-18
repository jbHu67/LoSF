from typing import Any

from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R


class LocalDataset(Dataset):
    def __init__(
        self,
        id_list: Any = None,
        data_dir: str = None,
        has_noise: bool = False,
        noise_level: float = 0.01,
        has_outliers: bool = False,
        outlier_ratio: float = 0.1,
        has_transform: bool = True,  # random rotation and translation
    ):
        self.id_list = id_list
        self.data_dir = data_dir
        self.has_noise = has_noise
        if self.has_noise:
            self.noise_level = noise_level
        self.has_outliers = has_outliers
        if self.has_outliers:
            self.outlier_ratio = outlier_ratio
        self.has_transform = has_transform

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, index):
        input_data = {
            "sample_id": None,
            "verts": None,
            "query": None,
            "vecs_q": None,
            "gt_udf": None,
        }
        """
        sample_id: id of the sample
        verts: points within local patch
        query: query point (a point per sample)
        vecs_q: vectors from query point to verts  
        gt_udf: ground truth udf value (a scalar per sample)  
        """
        sample_id = self.id_list[index]
        data_path = f"{self.data_dir}/{sample_id}.npz"
        data = np.load(data_path)
        gt_udf = data["z_height"]
        verts = data["verts"]
        query = data["query"]
        # add noise
        if self.has_noise:
            add_flag = np.random.uniform(0, 1)
            if add_flag >= 0.7:
                verts += np.random.normal(0, self.noise_level, verts.shape)
        # add outliers
        if self.has_outliers:
            add_flag = np.random.uniform(0, 1)
            if add_flag >= 0.7:
                num_outliers = int(len(verts) * self.outlier_ratio)
                outlier_idx = np.random.choice(len(verts), num_outliers, replace=False)
                verts[outlier_idx] += np.random.normal(0, 10, (num_outliers, 3))
        # add transform
        if self.has_transform:
            # # random 3d rotation
            # theta_x = np.random.uniform(0, 2 * np.pi)
            # theta_y = np.random.uniform(0, 2 * np.pi)
            # theta_z = np.random.uniform(0, 2 * np.pi)
            # RX = np.array(
            #     [
            #         [1, 0, 0],
            #         [0, np.cos(theta_x), -np.sin(theta_x)],
            #         [0, np.sin(theta_x), np.cos(theta_x)],
            #     ]
            # )
            # RY = np.array(
            #     [
            #         [np.cos(theta_y), 0, np.sin(theta_y)],
            #         [0, 1, 0],
            #         [-np.sin(theta_y), 0, np.cos(theta_y)],
            #     ]
            # )
            # RZ = np.array(
            #     [
            #         [np.cos(theta_z), -np.sin(theta_z), 0],
            #         [np.sin(theta_z), np.cos(theta_z), 0],
            #         [0, 0, 1],
            #     ]
            # )
            # R = np.dot(RZ, np.dot(RY, RX))
            random_quanternion = np.random.rand(4)
            random_quanternion /= np.linalg.norm(random_quanternion)
            rotation = R.from_quat(random_quanternion)
            R_matrix = rotation.as_matrix()
            verts = np.dot(verts, R_matrix)
            query = np.dot(query, R_matrix)

            # # random translation
            # translation_range = [-0.4, 0.4]
            # translation = np.random.uniform(
            #     translation_range[0], translation_range[1], 3
            # )
            # verts += translation
            # query += translation
        vecs_q = verts - query
        input_data["sample_id"] = sample_id
        input_data["verts"] = verts
        input_data["query"] = query
        input_data["vecs_q"] = vecs_q
        input_data["gt_udf"] = gt_udf
        return input_data
