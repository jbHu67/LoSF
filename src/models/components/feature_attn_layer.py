import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class CrossAttnModule(nn.Module):
    def __init__(self, pts_dim, vec_dim, out_dim):
        super(CrossAttnModule, self).__init__()
        self.query = nn.Linear(vec_dim, out_dim)
        self.key = nn.Linear(pts_dim, out_dim)
        self.value = nn.Linear(pts_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, pts, vec):
        q = self.query(vec)
        k = self.key(pts)
        v = self.value(pts)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim**0.5)
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)

        return out


class FeatureAttnLayer(nn.Module):
    def __init__(self, pts_dim, vec_dim, out_dim, num_self_attn=3):
        super(FeatureAttnLayer, self).__init__()
        self.cross_attn = CrossAttnModule(pts_dim, vec_dim, out_dim)
        self.denoise_attn = CrossAttnModule(pts_dim, pts_dim, out_dim)
        self.denoise_block = nn.Sequential(
            nn.Linear(out_dim + 128, out_dim + 128),
            nn.LeakyReLU(),
            nn.Linear(out_dim + 128, out_dim + 128),
            nn.LeakyReLU(),
            nn.Linear(out_dim + 128, 128),
        )
        # self.cross_attn = MultiheadAttention(embed_dim=pts_dim, num_heads=4)
        # activation function
        self.activation = nn.LeakyReLU()
        # self.self_attns = nn.ModuleList(
        #     [nn.MultiheadAttention(out_dim, 1) for _ in range(num_self_attn)]
        # )
        # define a set of fully connected layers
        self.FC = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
        )
        self.output_layer = nn.Linear(out_dim, 1)
        self.output_layer = nn.Sequential(
            nn.Linear(out_dim + 128, out_dim + 128),
            nn.LeakyReLU(),
            nn.Linear(out_dim + 128, 1),
            nn.Softplus(),
        )

    def forward(self, pts, vec, pts_denoise, distance):
        cross_attn_out = self.cross_attn(pts, vec)
        cross_attn_out = self.activation(cross_attn_out)
        self_attn_out = cross_attn_out
        self_attn_out = self.FC(self_attn_out)
        denoise_out = self.denoise_attn(pts_denoise, pts_denoise)
        denoise_out = torch.concat([denoise_out, distance], dim=1)
        displacement = self.denoise_block(denoise_out)
        # distance += displacement
        # displacement = torch.zeros_like(distance)
        self_attn_out = torch.concat([self_attn_out, distance], dim=1)
        out = self.output_layer(self_attn_out)
        return out, displacement
