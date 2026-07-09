"""
AASIST3 (Kolmogorov-Arnold Network Upgrade Implementation)
Copyright (c) 2026-present Your Portfolio / AASIST3 Team
MIT license
"""

import math
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kaggle environment me efficient-kan auto install setup
try:
    from efficient_kan import KAN
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "efficient-kan"])
    from efficient_kan import KAN

# ── 1. KAN GRAPH ATTENTION MODULES ───────────────────────────────────

class KAN_GraphAttentionLayer(nn.Module):
    """ARTICLE EQ 33 & 34: KAN Graph Attention Layer (KAN-GAL)"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = KAN([in_dim, out_dim])
        self.att_weight = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)

        self.proj_with_att = KAN([in_dim, out_dim])
        self.proj_without_att = KAN([in_dim, out_dim])
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        nb_nodes = x.size(1)

        x_exp = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        att_map = x_exp * x_exp.transpose(1, 2)

        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight) / self.temp
        att_map = F.softmax(att_map, dim=-2)

        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        out = x1 + x2

        org_size = out.size()
        out = self.bn(out.view(-1, org_size[-1])).view(org_size)
        return self.act(out)


class KAN_HtrgGraphAttentionLayer(nn.Module):
    """ARTICLE EQ 35 & 36: KAN Heterogeneous Stacked Graph Attention (KAN-HS-GAL)"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = KAN([in_dim, in_dim])
        self.proj_type2 = KAN([in_dim, in_dim])
        self.att_proj = KAN([in_dim, out_dim])
        self.att_projM = KAN([in_dim, out_dim])

        self.att_weight11 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.FloatTensor(out_dim, 1))
        for w in [self.att_weight11, self.att_weight22, self.att_weight12, self.att_weightM]:
            nn.init.xavier_normal_(w)

        self.proj_with_att = KAN([in_dim, out_dim])
        self.proj_without_att = KAN([in_dim, out_dim])
        self.proj_with_attM = KAN([in_dim, out_dim])
        self.proj_without_attM = KAN([in_dim, out_dim])

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)

        nb_nodes = x.size(1)
        x_exp = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        pair_mul = x_exp * x_exp.transpose(1, 2)
        att_map_raw = torch.tanh(self.att_proj(pair_mul))

        att_board = torch.zeros_like(att_map_raw[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(att_map_raw[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(att_map_raw[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(att_map_raw[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(att_map_raw[:, num_type1:, :num_type1, :], self.att_weight12)
        att_map = F.softmax(att_board / self.temp, dim=-2)

        att_map_M = F.softmax(torch.tanh(self.att_projM(x * master)) * self.att_weightM / self.temp, dim=-2)
        master = self.proj_with_attM(torch.matmul(att_map_M.squeeze(-1).unsqueeze(1), x)) + self.proj_without_attM(master)

        out = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        org_size = out.size()
        out = self.bn(out.view(-1, org_size[-1])).view(org_size)
        out = self.act(out)

        return out.narrow(1, 0, num_type1), out.narrow(1, num_type1, num_type2), master


class KAN_GraphPool(nn.Module):
    """Node pooling via KAN-scored top-k selection."""
    def __init__(self, k: float, in_dim: int, p: float):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = KAN([in_dim, 1])
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        Z = self.drop(h)
        scores = self.sigmoid(self.proj(Z))
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        return torch.gather(h * scores, 1, idx)

# ── 2. LEARNABLE FRONT-END (SincConv) ────────────────────────────────

class AASIST3SincConv(nn.Module):
    """Trainable parametric bandpass filter banks."""
    @staticmethod
    def to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def to_hz(mel): return 700 * (10**(mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.sample_rate = sample_rate

        freqs = int(self.sample_rate / 2) * np.linspace(0, 1, int(512 / 2) + 1)
        filbandwidthsf = self.to_hz(np.linspace(np.min(self.to_mel(freqs)), np.max(self.to_mel(freqs)), out_channels + 1))

        self.mel_low = nn.Parameter(torch.from_numpy(filbandwidthsf[:-1]).float())
        self.mel_band = nn.Parameter(torch.from_numpy(np.diff(filbandwidthsf)).float())

        self.register_buffer("hsupp", torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1).float())
        self.register_buffer("window", torch.from_numpy(np.hamming(self.kernel_size).astype(np.float32)))

    def forward(self, x):
        low = torch.clamp(self.mel_low, min=0, max=self.sample_rate / 2)
        band = torch.clamp(self.mel_band, min=0, max=self.sample_rate / 2 - low)
        high = low + band

        filters = []
        for i in range(self.out_channels):
            fmin, fmax = low[i], high[i]
            hHigh = (2 * fmax / self.sample_rate) * torch.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * torch.sinc(2 * fmin * self.hsupp / self.sample_rate)
            filters.append((hHigh - hLow) * self.window)

        return F.conv1d(x.unsqueeze(1) if x.dim() == 2 else x, torch.stack(filters).unsqueeze(1), stride=1, padding=0)

# ── 3. MAIN AASIST3 PIPELINE ASSEMBLY ────────────────────────────────

class Model(nn.Module):
    """
    AASIST3 Complete Network Architecture Upgrade.
    Replaces 2D CNN encoders with the SincConv pre-encoder, and deploys
    4 parallel branches of KAN-HS-GAL layers stacking.
    """
    def __init__(self, d_args):
        super().__init__()
        feature_dim = d_args["gat_dims"][0]
        out_channels = d_args["filts"][0]

        # Front-End pipeline: SincConv -> MaxPool -> BatchNorm -> SELU
        self.conv_time = AASIST3SincConv(out_channels=out_channels, kernel_size=d_args["first_conv"])
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.selu = nn.SELU(inplace=True)

        # FIX: compute the actual post-frontend time length instead of
        # hardcoding a number that only matches one specific kernel_size.
        # BatchNorm/SELU don't change the time dimension, so we can skip
        # them here and avoid batch-size-1 BatchNorm issues.
        with torch.no_grad():
            dummy = torch.zeros(1, 64600)
            dummy_feat = self.max_pool(self.conv_time(dummy))
            reduced_time = dummy_feat.shape[-1]

        self.temporal_proj = nn.Linear(reduced_time, feature_dim)
        self.spatial_proj = nn.Linear(out_channels, feature_dim)

        # Learnable Stack Node Token (S1)
        self.S1 = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Initial Spatial/Temporal KAN Graph Attention Layers (Eq 33, 34)
        self.GAT_layer_T = KAN_GraphAttentionLayer(feature_dim, feature_dim, temperature=d_args["temperatures"][0])
        self.GAT_layer_S = KAN_GraphAttentionLayer(feature_dim, feature_dim, temperature=d_args["temperatures"][1])
        self.pool_T = KAN_GraphPool(d_args["pool_ratios"][0], feature_dim, 0.3)
        self.pool_S = KAN_GraphPool(d_args["pool_ratios"][1], feature_dim, 0.3)

        # 4 parallel branches (Eq 35 & 36 blocks)
        self.kan_hs_gal1 = nn.ModuleList([KAN_HtrgGraphAttentionLayer(feature_dim, feature_dim) for _ in range(4)])
        self.temporal_branch_pool = nn.ModuleList([KAN_GraphPool(d_args["pool_ratios"][2], feature_dim, 0.3) for _ in range(4)])
        self.spatial_branch_pool = nn.ModuleList([KAN_GraphPool(d_args["pool_ratios"][2], feature_dim, 0.3) for _ in range(4)])
        self.kan_hs_gal2 = nn.ModuleList([KAN_HtrgGraphAttentionLayer(feature_dim, feature_dim) for _ in range(4)])

        self.drop_branch = nn.Dropout(p=0.2)
        self.drop_readout = nn.Dropout(p=0.5)
        self.out_layer = KAN([5 * feature_dim, 2])

    def forward(self, x, Freq_aug=False):
        # 1. Frontend
        x_feats = self.conv_time(x)
        x_pooled = self.max_pool(x_feats)
        x_norm = self.batch_norm(x_pooled)
        x_hat = self.selu(x_norm)  # [Batch, Out_Channels, Reduced_Time]

        # 2. Axis splits and graph transforms
        t_base = self.temporal_proj(x_hat)                 # [Batch, Out_Channels, feature_dim]
        s_base = self.spatial_proj(x_hat.transpose(1, 2))  # [Batch, Reduced_Time, feature_dim]

        ht1 = self.pool_T(self.GAT_layer_T(t_base))
        hs1 = self.pool_S(self.GAT_layer_S(s_base))

        batch_size = x.size(0)
        S1 = self.S1.expand(batch_size, -1, -1)

        Ht = torch.zeros_like(ht1)
        Hs = torch.zeros_like(hs1)
        Sf = torch.zeros_like(S1)

        # 3. Run the 4 parallel branches
        for i in range(4):
            ht2, hs2, S2 = self.kan_hs_gal1[i](ht1, hs1, S1)
            ht2_p = self.temporal_branch_pool[i](ht2)
            hs2_p = self.spatial_branch_pool[i](hs2)
            ht3, hs3, S3 = self.kan_hs_gal2[i](ht2_p, hs2_p, S2)

            Ht = Ht + ht1 + ht2_p + ht3
            Hs = Hs + hs1 + hs2_p + hs3
            Sf = Sf + S1 + S2 + S3

        Ht, Hs, Sf = self.drop_branch(Ht), self.drop_branch(Hs), self.drop_branch(Sf)

        # 4. Statistical pooling and concatenation
        Hmax_t, _ = torch.max(torch.abs(Ht), dim=1)
        Hmean_t = torch.mean(Ht, dim=1)
        Hmax_s, _ = torch.max(torch.abs(Hs), dim=1)
        Hmean_s = torch.mean(Hs, dim=1)
        Smax_f, _ = torch.max(torch.abs(Sf), dim=1)

        Hmax_t = self.drop_readout(Hmax_t)
        Hmean_t = self.drop_readout(Hmean_t)
        Hmax_s = self.drop_readout(Hmax_s)
        Hmean_s = self.drop_readout(Hmean_s)
        Smax_f = self.drop_readout(Smax_f)

        L = torch.cat([Hmax_t, Hmean_t, Hmax_s, Hmean_s, Smax_f], dim=1)
        output = self.out_layer(L)
        return L, output
