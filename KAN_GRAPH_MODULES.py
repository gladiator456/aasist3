import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# KAN implementation support import kiya
try:
    from efficient_kan import KAN
except ImportError:
    print("Please install efficient-kan: pip install efficient-kan")
    sys.exit()

class KAN_GraphAttentionLayer(nn.Module):
    """
    ARTICLE EQ 33 & 34: KAN-GAL (KAN Graph Attention Layer)
    """
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        # ARTICLE UPDATE: Linear layers ki jagah KAN activation grids setup kiye
        self.att_proj = KAN([in_dim, out_dim])
        self.att_weight = self._init_new_params(out_dim, 1)
        
        self.proj_with_att = KAN([in_dim, out_dim])
        self.proj_without_att = KAN([in_dim, out_dim])
        
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        # KAN based dynamic projections
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class KAN_HtrgGraphAttentionLayer(nn.Module):
    """
    ARTICLE EQ 35 & 36: KAN-HS-GAL (KAN Heterogeneous Stacked Graph Attention Layer)
    Yeh handle karega Temporal Graph (x1), Spatial Graph (x2) aur Stack Node (master).
    """
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        # ARTICLE UPDATE: Saare dense projection steps KAN mapping se chalenge
        self.proj_type1 = KAN([in_dim, in_dim])
        self.proj_type2 = KAN([in_dim, in_dim])
        
        self.att_proj = KAN([in_dim, out_dim])
        self.att_projM = KAN([in_dim, out_dim])
        
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        
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
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        
        # Heterogeneous split fir se narrow matrix slicing se kiya
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)
            
        att_map = att_board
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class KAN_GraphPool(nn.Module):
    """
    ARTICLE CORRECTION: KAN-GraphPool layer 
    Top-k nodes features pooling process controlled via KAN learning matrix scores.
    """
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        # ARTICLE UPDATE: Linear scoring line replaced with KAN mapping grid
        self.proj = KAN([in_dim, 1])
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h
