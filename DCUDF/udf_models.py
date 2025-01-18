import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import pytorch3d.ops
import pytorch3d


class UDF(nn.Module):
    def __init__(self, K=10):
        super(UDF, self).__init__()

        self.attention_net = nn.Sequential(
            nn.Conv2d(6 + 1 + 128 + 128, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 1, 1),
        )

        self.grad_attention_net = nn.Sequential(
            nn.Conv2d(6 + 1 + 128 + 128, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 1, 1),
        )
        self.K = K

        self.patch_feature_net = nn.Sequential(
            nn.Conv2d(6, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, 1),
        )

    def forward(self, input_pcd, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        query_points: query points [x, y, z] with shape: (B, Num_points, 3)
        """

        raise NotImplementedError


class Weighted_Dist_UDF(UDF):
    def __init__(self, K=10, norm=2, mode="None"):
        super(Weighted_Dist_UDF, self).__init__()
        self.K = K
        self.norm = norm
        self.mode = mode

    def forward(self, input_pcd, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        query_points: query points [x, y, z] with shape: (B, 3, Num_points)
        """

        query_points = query_points.transpose(1, 2)

        _, idx, query_knn_pc = pytorch3d.ops.knn_points(
            query_points, input_pcd, K=self.K, return_nn=True, return_sorted=False
        )  # (B, Num_points, K) (B, Num_points, K, 3)

        query_knn_pc_local = (
            query_points.unsqueeze(2) - query_knn_pc
        )  # (B, Num_points, K, 3)

        k_distance = torch.abs(
            torch.norm(query_knn_pc_local, p=self.norm, dim=-1, keepdim=True)
        )  # (B, Num_points, K ,1)

        concat_vector_1 = torch.concat(
            (query_knn_pc_local, query_points.unsqueeze(2).repeat(1, 1, 10, 1)), dim=3
        ).permute(
            0, 3, 1, 2
        )  # (B, Num_points, K, 6)->(B, 6, Num_points, K)

        feature = self.patch_feature_net(concat_vector_1)  # (B, 128, M, K)
        patch_feature = torch.max(feature, dim=3, keepdim=True)[
            0
        ]  # (B, 128, Num_points, 1)

        concat_vector_2 = torch.concat(
            (
                concat_vector_1,
                k_distance.permute(0, 3, 1, 2),
                feature,
                patch_feature.repeat(1, 1, 1, self.K),
            ),
            dim=1,
        )

        weights = self.attention_net(concat_vector_2).squeeze(
            1
        )  # (B, 1, Num_points, K)->(B, Num_points, K)
        weights = F.softmax(weights, dim=2)  # (B, Num_points, K)

        udf = torch.norm(
            torch.sum(weights.unsqueeze(-1) * query_knn_pc_local, dim=-2),
            p=2,
            dim=-1,
            keepdim=True,
        ).squeeze(
            -1
        )  # (B, Num_points)

        if self.mode == "debug":
            estimated_vector = torch.sum(
                weights.unsqueeze(-1) * query_knn_pc_local, dim=-2
            )

            return udf, estimated_vector, weights, query_knn_pc_local

        return udf


class Weighted_Dist_Displacement_UDF(UDF):
    def __init__(self, K=10, norm=2, mode="None"):
        super(Weighted_Dist_Displacement_UDF, self).__init__()
        self.K = K
        self.norm = norm
        self.mode = mode

        self.attention_net = nn.Sequential(
            nn.Conv2d(6 + 1 + 128 + 128, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1),
        )

    def forward(self, input_pcd, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        query_points: query points [x, y, z] with shape: (B, 3, Num_points)
        """

        query_points = query_points.transpose(1, 2)

        _, idx, query_knn_pc = pytorch3d.ops.ball_query(
            query_points, input_pcd, K=self.K, radius=0.02, return_nn=True
        )  # (B, Num_points, K) (B, Num_points, K, 3)

        query_knn_pc_local = (
            query_points.unsqueeze(2) - query_knn_pc
        )  # (B, Num_points, K, 3)

        k_distance = torch.abs(
            torch.norm(query_knn_pc_local, p=self.norm, dim=-1, keepdim=True)
        )  # (B, Num_points, K ,1)

        concat_vector_1 = torch.concat(
            (query_knn_pc_local, query_points.unsqueeze(2).repeat(1, 1, self.K, 1)),
            dim=3,
        ).permute(
            0, 3, 1, 2
        )  # (B, Num_points, K, 6)->(B, 6, Num_points, K)

        feature = self.patch_feature_net(concat_vector_1)  # (B, 128, M, K)
        patch_feature = torch.max(feature, dim=3, keepdim=True)[
            0
        ]  # (B, 128, Num_points, 1)

        concat_vector_2 = torch.concat(
            (
                concat_vector_1,
                k_distance.permute(0, 3, 1, 2),
                feature,
                patch_feature.repeat(1, 1, 1, self.K),
            ),
            dim=1,
        )

        attention_result = self.attention_net(concat_vector_2)  # (B, 4, Num_points, K)
        weights = attention_result[:, 0, :, :].squeeze(1)  # (B, Num_points, K)
        displacement = attention_result[:, 1:, :, :].permute(
            0, 2, 3, 1
        )  # (B, 3, Num_points, K) -> (B, Num, K, 3)
        weights = F.softmax(weights, dim=2)  # (B, Num_points, K)

        udf = torch.norm(
            torch.sum(
                weights.unsqueeze(-1) * (query_knn_pc_local - displacement), dim=-2
            ),
            p=2,
            dim=-1,
            keepdim=True,
        ).squeeze(
            -1
        )  # (B, Num_points)

        if self.mode == "debug":
            estimated_vector = torch.sum(
                weights.unsqueeze(-1) * query_knn_pc_local, dim=-2
            )

            return udf, estimated_vector, weights, query_knn_pc_local

        return udf


class Dist_Weighted_UDF(UDF):
    def __init__(self, K=10, norm=2):
        super(Dist_Weighted_UDF, self).__init__()
        self.K = K
        self.norm = norm

    def forward(self, input_pcd, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        query_points: query points [x, y, z] with shape: (B, Num_points, 3)
        """

        _, idx, query_knn_pc = pytorch3d.ops.knn_points(
            query_points, input_pcd, K=self.K, return_nn=True, return_sorted=False
        )  # (B, Num_points, K) (B, Num_points, K, 3)

        query_knn_pc_local = (
            query_points.unsqueeze(2) - query_knn_pc
        )  # (B, Num_points, K, 3)

        k_distance = torch.abs(
            torch.norm(query_knn_pc_local, p=1, dim=-1, keepdim=True)
        )  # (B, Num_points, K, 1)

        concat_vector_1 = torch.concat(
            (query_knn_pc_local, query_points.unsqueeze(2).repeat(1, 1, 10, 1)), dim=3
        ).permute(
            0, 3, 1, 2
        )  # (B, Num_points, K, 6)->(B, 6, Num_points, K)

        feature = self.patch_feature_net(concat_vector_1)  # (B, 128, Num_points, K)
        patch_feature = torch.max(feature, dim=3, keepdim=True)[
            0
        ]  # (B, 128, Num_points, 1)

        concat_vector_2 = torch.concat(
            (
                concat_vector_1,
                k_distance.permute(0, 3, 1, 2),
                feature,
                patch_feature.repeat(1, 1, 1, self.K),
            ),
            dim=1,
        )

        weights = self.attention_net(concat_vector_2).squeeze(
            1
        )  # (B, 1, Num_points, K)->(B, Num_points, K)
        weights = F.softmax(weights, dim=2)

        udf = torch.sum(weights * k_distance.squeeze(3), dim=2)  # (B, Num_points)

        return udf


class Normal_Weighted_UDF(UDF):
    """
    This is the implementation for UDF model in GeoUDF but without gradient calculation
    """

    def __init__(self, K=10, norm=2):
        super(Normal_Weighted_UDF, self).__init__()
        self.K = K
        self.norm = norm

    def forward(self, input_pcd, input_normal, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        input_normal: normal of the input point cloud
        query_points: query points [x, y, z] with shape: (B, Num_points, 3)
        """

        _, idx, query_knn_pc = pytorch3d.ops.knn_points(
            query_points.transpose(1, 2),
            input_pcd.transpose(1, 2),
            K=self.K,
            return_nn=True,
            return_sorted=False,
        )  # (B, Num_points, K) (B, Num_points, K, 3)

        query_knn_normal = pytorch3d.ops.knn_gather(input_normal)

        query_knn_pc_local = (
            query_points.transpose(1, 2).unsqueeze(2) - query_knn_pc
        )  # (B, Num_points, K, 3)

        k_signed_distance = torch.sum(
            query_knn_pc_local * query_knn_normal, dim=3, keepdim=True
        )  # (B, M, K, 1)

        k_distance = torch.abs(k_signed_distance)
        original_normal = torch.sgn(k_signed_distance) * query_knn_normal

        concat_vector_1 = torch.concat(
            (
                query_knn_pc_local,
                query_points.transpose(1, 2).unsqueeze(2).repeat(1, 1, 10, 1),
            ),
            dim=3,
        ).permute(
            0, 3, 1, 2
        )  # (B, Num_points, K, 6)->(B, 6, Num_points, K)

        feature = self.patch_feature_net(concat_vector_1)  # (B, 128, M, K)
        patch_feature = torch.max(feature, dim=3, keepdim=True)[
            0
        ]  # (B, 128, Num_points, 1)

        concat_vector_2 = torch.concat(
            (
                concat_vector_1,
                k_distance.permute(0, 3, 1, 2),
                feature,
                patch_feature.repeat(1, 1, 1, self.K),
            ),
            dim=1,
        )

        weights = self.attention_net(concat_vector_2).squeeze(
            1
        )  # (B, 1, Num_points, K)->(B, Num_points, K)
        weights = F.softmax(weights, dim=2)  # (B, Num_points, K)

        udf = torch.norm(
            torch.sum(weights.unsqueeze(-1) * query_knn_pc_local, dim=-2),
            p=2,
            dim=-1,
            keepdim=True,
        ).squeeze(
            -1
        )  # (B, Num_points)

        if self.mode == "debug":
            estimated_vector = torch.sum(
                weights.unsqueeze(-1) * query_knn_pc_local, dim=-2
            )

            return udf, estimated_vector

        return udf


class Average_Weighted_UDF(UDF):
    def __init__(self, K=10, norm=2, mode="None"):
        super(Average_Weighted_UDF, self).__init__()
        self.K = K
        self.norm = norm
        self.mode = mode

    def forward(self, input_pcd, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        query_points: query points [x, y, z] with shape: (B, 3, Num_points)
        """

        query_points = query_points.transpose(1, 2)

        _, idx, query_knn_pc = pytorch3d.ops.knn_points(
            query_points, input_pcd, K=self.K, return_nn=True, return_sorted=False
        )  # (B, Num_points, K) (B, Num_points, K, 3)

        query_knn_pc_local = (
            query_points.unsqueeze(2) - query_knn_pc
        )  # (B, Num_points, K, 3)

        udf = torch.norm(
            torch.sum(0.1 * query_knn_pc_local, dim=-2), p=2, dim=-1, keepdim=True
        ).squeeze(-1)

        return udf


class Weighted_BNN_UDF(UDF):
    """
    This is a clean Ball query Weighted UDF model with no displacement
    """

    def __init__(self, K=10, norm=2, mode="None"):
        super(Weighted_BNN_UDF, self).__init__()
        self.K = K
        self.norm = norm
        self.mode = mode

        self.displacement_attention_net = None
        self.grad_attention_net = None

    def forward(self, input_pcd, query_points):
        """
        input_pcd: original point cloud [x, y, z] with shape: (B, Num_points, 3)
        query_points: query points [x, y, z] with shape: (B, 3, Num_points)
        """

        query_points = query_points.transpose(1, 2)

        _, idx, query_knn_pc_orinal = pytorch3d.ops.ball_query(
            query_points, input_pcd, K=self.K, radius=0.1, return_nn=True
        )  # (B, Num_points, K) (B, Num_points, K, 3)

        # We should replace all the Ball neighbors which are invalid, i.e. idx = -1
        valid_idx = idx != -1  # (B, Num_points, K)
        invalid_idx = idx == -1  # (B, Num_points, K)
        valid_count = torch.sum(valid_idx, dim=-1)  # (B, Num_points)

        query_knn_pc_sum = torch.sum(query_knn_pc_orinal, dim=2)  # (B, Num_point, 3)

        invalid_query_points_idx = (valid_count == 0).nonzero()
        valid_query_points_idx = (valid_count != 0).nonzero()

        # Remove all the zero valid idx cases to 1:
        valid_count_filtered = torch.where(
            valid_count == 0, torch.ones_like(valid_count), valid_count
        )

        query_knn_pc_sum = torch.sum(query_knn_pc_orinal, dim=2)  # (B, Num_point, 3)
        query_knn_pc_mean = query_knn_pc_sum / valid_count_filtered.unsqueeze(
            -1
        )  # (B, Num_point, 3)

        # Apply this mean value to all invalid
        query_knn_pc = query_knn_pc_orinal + query_knn_pc_mean.unsqueeze(
            2
        ) * invalid_idx.unsqueeze(-1)

        query_knn_pc_local = (
            query_points.unsqueeze(2) - query_knn_pc
        )  # (B, Num_points, K, 3)

        k_distance = torch.abs(
            torch.norm(query_knn_pc_local, p=self.norm, dim=-1, keepdim=True)
        )  # (B, Num_points, K ,1)

        concat_vector_1 = torch.concat(
            (query_knn_pc, query_points.unsqueeze(2).repeat(1, 1, self.K, 1)),
            dim=3,
        ).permute(
            0, 3, 1, 2
        )  # (B, Num_points, K, 6)->(B, 6, Num_points, K)

        feature = self.patch_feature_net(concat_vector_1)  # (B, 128, M, K)
        patch_feature = torch.max(feature, dim=3, keepdim=True)[
            0
        ]  # (B, 128, Num_points, 1)

        concat_vector_2 = torch.concat(
            (
                concat_vector_1,
                k_distance.permute(0, 3, 1, 2),
                feature,
                patch_feature.repeat(1, 1, 1, self.K),
            ),
            dim=1,
        )

        weights = self.attention_net(concat_vector_2).squeeze(
            1
        )  # (B, 1, Num_points, K)->(B, Num_points, K)
        weights = F.softmax(weights, dim=2)  # (B, Num_points, K)

        udf = torch.norm(
            torch.sum(weights.unsqueeze(-1) * query_knn_pc_local, dim=-2),
            p=2,
            dim=-1,
            keepdim=True,
        ).squeeze(
            -1
        )  # (B, Num_points)

        return udf, valid_query_points_idx, valid_count
