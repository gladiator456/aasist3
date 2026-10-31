"""KAN-based audio anti-spoofing model following the AASIST3 paper's closed-condition design.

Pre-emphasis -> fixed SincConv -> abs/max-pool -> BN -> SELU -> 6-block residual
encoder -> spatial (frequency) and temporal (time) graphs with positional
embeddings -> KAN-GAL / KAN-GraphPool -> 4 KAN-HS-GAL branches -> KAN classifier.
The graphs are small (about 23 spatial and 29 temporal nodes), so dense pairwise
attention stays cheap. This is a research prototype, not a guaranteed
reproduction of the published results.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KAN


class KAN_GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0, dropout=0.2):
        super().__init__()
        self.att_proj = KAN([in_dim, out_dim])
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_with_att = KAN([in_dim, out_dim])
        self.proj_without_att = KAN([in_dim, out_dim])
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(dropout)
        self.act = nn.SELU(inplace=True)
        self.temp = max(float(temperature), 1e-4)

    def forward(self, x):
        # x: [batch, nodes, features]. Nodes must remain bounded.
        x = self.input_drop(x)
        n = x.size(1)
        pair = x.unsqueeze(2) * x.unsqueeze(1)  # [B,N,N,D]
        scores = torch.matmul(torch.tanh(self.att_proj(pair)), self.att_weight)
        att = F.softmax(scores / self.temp, dim=2).squeeze(-1)  # [B,N,N]
        out = self.proj_with_att(torch.matmul(att, x)) + self.proj_without_att(x)
        b, n, d = out.shape
        out = self.bn(out.reshape(-1, d)).reshape(b, n, d)
        return self.act(out)


class KAN_HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous graph attention with a correctly shaped master-node update."""
    def __init__(self, in_dim, out_dim, temperature=1.0, dropout=0.2):
        super().__init__()
        self.proj_type1 = KAN([in_dim, in_dim])
        self.proj_type2 = KAN([in_dim, in_dim])
        self.att_proj = KAN([in_dim, out_dim])
        self.att_projM = KAN([in_dim, out_dim])
        self.att_weight11 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.empty(out_dim, 1))
        for p in (self.att_weight11, self.att_weight22, self.att_weight12, self.att_weightM):
            nn.init.xavier_normal_(p)
        self.proj_with_att = KAN([in_dim, out_dim])
        self.proj_without_att = KAN([in_dim, out_dim])
        self.proj_with_attM = KAN([in_dim, out_dim])
        self.proj_without_attM = KAN([in_dim, out_dim])
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(dropout)
        self.act = nn.SELU(inplace=True)
        self.temp = max(float(temperature), 1e-4)

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x1, x2 = self.proj_type1(x1), self.proj_type2(x2)
        x = self.input_drop(torch.cat([x1, x2], dim=1))
        if master is None:
            master = x.mean(dim=1, keepdim=True)

        pair = x.unsqueeze(2) * x.unsqueeze(1)
        pair_scores = torch.tanh(self.att_proj(pair))
        scores = torch.zeros_like(pair_scores[..., :1])
        scores[:, :n1, :n1] = pair_scores[:, :n1, :n1] @ self.att_weight11
        scores[:, n1:, n1:] = pair_scores[:, n1:, n1:] @ self.att_weight22
        scores[:, :n1, n1:] = pair_scores[:, :n1, n1:] @ self.att_weight12
        scores[:, n1:, :n1] = pair_scores[:, n1:, :n1] @ self.att_weight12
        att = F.softmax(scores / self.temp, dim=2).squeeze(-1)
        out = self.proj_with_att(att @ x) + self.proj_without_att(x)

        master_features = torch.tanh(self.att_projM(x * master))
        master_scores = (master_features @ self.att_weightM) / self.temp  # [B,N,1]
        master_att = F.softmax(master_scores, dim=1)
        master_update = self.proj_with_attM(master_att.transpose(1, 2) @ x)
        master = master_update + self.proj_without_attM(master)

        b, n, d = out.shape
        out = self.bn(out.reshape(-1, d)).reshape(b, n, d)
        out = self.act(out)
        return out[:, :n1], out[:, n1:n1+n2], master


class KAN_GraphPool(nn.Module):
    def __init__(self, k, in_dim, p=0.0):
        super().__init__()
        self.k = float(k)
        self.proj = KAN([in_dim, 1])
        self.drop = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, h):
        scores = torch.sigmoid(self.proj(self.drop(h)))
        n_keep = max(1, min(h.size(1), int(round(h.size(1) * self.k))))
        idx = torch.topk(scores.squeeze(-1), n_keep, dim=1).indices.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return torch.gather(h * scores, 1, idx)


TARGET_SAMPLES = 64600  # 16 kHz, ~4.04 s: the fixed input length the model is built for

# Single source of truth for the model configuration; TRAIN.py, SMOKE_TEST.py,
# EVAL.py and MODELS.py all import this so the settings cannot drift apart.
# "arch" and "pre_emphasis" also make checkpoints from the older encoder-less
# model fail the d_args comparison with a clear error instead of loading wrongly.
DEFAULT_D_ARGS = {
    "arch": "aasist3-encoder-v2",
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 1.0, 1.0],
    "pre_emphasis": 0.97,
}


class AASIST3SincConv(nn.Module):
    """Fixed (non-trainable) mel-initialised band-pass filter bank.

    The AASIST3 paper uses a non-trainable SincConv front end. The filters are
    built once here instead of being recomputed on every forward pass.
    """
    @staticmethod
    def to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self.sample_rate = int(sample_rate)
        nyquist = self.sample_rate / 2
        hz = np.linspace(0, nyquist, 257)
        mel_edges = np.linspace(self.to_mel(hz).min(), self.to_mel(hz).max(), self.out_channels + 1)
        edges = self.to_hz(mel_edges)
        low = torch.tensor(edges[:-1], dtype=torch.float32).clamp(0, nyquist - 1)
        band = torch.tensor(np.diff(edges), dtype=torch.float32).clamp(1, nyquist)
        high = torch.minimum(low + band, torch.full_like(low, nyquist))
        t = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1).float()
        window = torch.from_numpy(np.hamming(self.kernel_size).astype(np.float32))
        hi_kernel = (2 * high[:, None] / self.sample_rate) * torch.sinc(2 * high[:, None] * t[None, :] / self.sample_rate)
        lo_kernel = (2 * low[:, None] / self.sample_rate) * torch.sinc(2 * low[:, None] * t[None, :] / self.sample_rate)
        self.register_buffer("kernels", ((hi_kernel - lo_kernel) * window).unsqueeze(1))  # [C,1,K]

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"SincConv expects [B,T] or [B,1,T], got {tuple(x.shape)}")
        return F.conv1d(x, self.kernels.to(dtype=x.dtype))


class ResidualBlock(nn.Module):
    """Encoder unit from the paper: ConvUnit(x) = Conv(SELU(BatchNorm(x))), applied
    twice, with a (down-sampled if needed) skip connection and max-pooling over time."""

    def __init__(self, nb_filts, first=False):
        super().__init__()
        in_ch, out_ch = int(nb_filts[0]), int(nb_filts[1])
        self.first = first
        self.bn1 = None if first else nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(2, 3), padding=(0, 1))
        self.selu = nn.SELU(inplace=True)
        self.conv_downsample = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), padding=(0, 1)) if in_ch != out_ch else None
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.selu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.conv_downsample is not None:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


def _align_nodes(x, n_nodes):
    """Resize the node axis so residual aggregation has matching dimensions."""
    if x.size(1) == n_nodes:
        return x
    return F.adaptive_avg_pool1d(x.transpose(1, 2), n_nodes).transpose(1, 2)


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        filts = d_args["filts"]
        feature_dim = int(d_args["gat_dims"][0])
        if int(filts[4][1]) != feature_dim:
            raise ValueError(f"Encoder output channels ({filts[4][1]}) must equal gat_dims[0] ({feature_dim}).")
        self.pre_emphasis = float(d_args.get("pre_emphasis", 0.97))

        # Front end: pre-emphasis -> SincConv -> abs -> 2-D max-pool -> BN -> SELU -> encoder.
        self.conv_time = AASIST3SincConv(int(filts[0]), d_args["first_conv"])
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            ResidualBlock(filts[1], first=True),
            ResidualBlock(filts[2]),
            ResidualBlock(filts[3]),
            ResidualBlock(filts[4]),
            ResidualBlock(filts[4]),
            ResidualBlock(filts[4]),
        )

        # Learned positional embeddings for the spatial (frequency) and temporal
        # (time) node sets. Node counts follow from the fixed input length; the
        # dummy pass runs in eval mode so BatchNorm statistics are not touched.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            encoded = self._encode(torch.zeros(1, TARGET_SAMPLES))
        self.train(was_training)
        n_freq, n_time = int(encoded.size(2)), int(encoded.size(3))
        self.pos_S = nn.Parameter(torch.randn(1, n_freq, feature_dim))
        self.pos_T = nn.Parameter(torch.randn(1, n_time, feature_dim))
        self.S1 = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        temps = d_args.get("temperatures", [1.0, 1.0])
        pools = d_args.get("pool_ratios", [0.4, 0.5, 0.7, 0.5])
        self.GAT_layer_T = KAN_GraphAttentionLayer(feature_dim, feature_dim, temps[0])
        self.GAT_layer_S = KAN_GraphAttentionLayer(feature_dim, feature_dim, temps[1])
        self.pool_T = KAN_GraphPool(pools[0], feature_dim, 0.3)
        self.pool_S = KAN_GraphPool(pools[1], feature_dim, 0.3)
        self.kan_hs_gal1 = nn.ModuleList([KAN_HtrgGraphAttentionLayer(feature_dim, feature_dim, temps[2] if len(temps) > 2 else 1.0) for _ in range(4)])
        self.temporal_branch_pool = nn.ModuleList([KAN_GraphPool(pools[2], feature_dim, 0.3) for _ in range(4)])
        self.spatial_branch_pool = nn.ModuleList([KAN_GraphPool(pools[2], feature_dim, 0.3) for _ in range(4)])
        self.kan_hs_gal2 = nn.ModuleList([KAN_HtrgGraphAttentionLayer(feature_dim, feature_dim, temps[3] if len(temps) > 3 else 1.0) for _ in range(4)])
        self.drop_branch = nn.Dropout(0.2)
        self.drop_readout = nn.Dropout(0.5)
        # The pooled readout is normalised so it sits inside the KAN spline grid;
        # without this, ~1/3 of classifier inputs fell outside [-1, 1] and the
        # initial logits were huge (std ~5, loss ~4-20 instead of ~0.7).
        self.readout_norm = nn.LayerNorm(5 * feature_dim)
        self.out_layer = KAN([5 * feature_dim, 2], grid_range=[-3, 3])

    def _encode(self, x):
        """Raw waveform [B, T] -> encoder feature map [B, C, freq_nodes, time_nodes]."""
        if self.pre_emphasis > 0:
            x = torch.cat([x[:, :1], x[:, 1:] - self.pre_emphasis * x[:, :-1]], dim=1)
        x = self.conv_time(x).unsqueeze(1)           # [B, 1, 70, T']
        x = F.max_pool2d(torch.abs(x), (3, 3))        # [B, 1, 23, T'/3]
        x = self.selu(self.first_bn(x))
        return self.encoder(x)                        # [B, 24, 23, ~29]

    def forward(self, x, Freq_aug=False):
        # Freq_aug retained for compatibility; no frequency augmentation is applied.
        if x.dim() != 2 or x.size(1) != TARGET_SAMPLES:
            raise ValueError(f"Model expects waveforms of shape [B, {TARGET_SAMPLES}], got {tuple(x.shape)}")
        e = torch.abs(self._encode(x))
        spatial = e.max(dim=3).values.transpose(1, 2) + self.pos_S   # [B, F, D]  (max over time)
        temporal = e.max(dim=2).values.transpose(1, 2) + self.pos_T  # [B, T'', D] (max over frequency)
        ht1 = self.pool_T(self.GAT_layer_T(temporal))
        hs1 = self.pool_S(self.GAT_layer_S(spatial))
        master1 = self.S1.expand(x.size(0), -1, -1)

        Ht = torch.zeros_like(ht1)
        Hs = torch.zeros_like(hs1)
        Sf = torch.zeros_like(master1)
        for i in range(4):
            ht2, hs2, master2 = self.kan_hs_gal1[i](ht1, hs1, master1)
            ht2p = self.temporal_branch_pool[i](ht2)
            hs2p = self.spatial_branch_pool[i](hs2)
            ht3, hs3, master3 = self.kan_hs_gal2[i](ht2p, hs2p, master2)
            Ht = Ht + ht1 + _align_nodes(ht2p, ht1.size(1)) + _align_nodes(ht3, ht1.size(1))
            Hs = Hs + hs1 + _align_nodes(hs2p, hs1.size(1)) + _align_nodes(hs3, hs1.size(1))
            Sf = Sf + master1 + master2 + master3

        # Average (not sum) over the 4 branches x 3 residual terms so the scale
        # of the readout does not grow with the number of branches.
        n_terms = 4 * 3
        Ht, Hs, Sf = Ht / n_terms, Hs / n_terms, Sf / n_terms
        Ht, Hs, Sf = self.drop_branch(Ht), self.drop_branch(Hs), self.drop_branch(Sf)
        pooled = [
            torch.max(torch.abs(Ht), dim=1).values,
            Ht.mean(dim=1),
            torch.max(torch.abs(Hs), dim=1).values,
            Hs.mean(dim=1),
            torch.max(torch.abs(Sf), dim=1).values,
        ]
        L = self.readout_norm(self.drop_readout(torch.cat(pooled, dim=1)))
        return L, self.out_layer(L)
