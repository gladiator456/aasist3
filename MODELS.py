import torch
import torch.nn as nn
import torch.nn.functional as F
from core.frontend import AASIST3FrontEndClosed
from core.kan_graph_modules import KAN_HtrgGraphAttentionLayer, KAN_GraphPool
from efficient_kan import KAN

class AASIST3_Branch_Block(nn.Module):
    """
    ARTICLE EQ 35 & 36: One single branch pipeline
    """
    def __init__(self, feature_dim=64, pool_ratio=0.5):
        super().__init__()
        self.kan_hs_gal1 = KAN_HtrgGraphAttentionLayer(in_dim=feature_dim, out_dim=feature_dim)
        self.temporal_pool = KAN_GraphPool(k=pool_ratio, in_dim=feature_dim, p=0.2)
        self.spatial_pool = KAN_GraphPool(k=pool_ratio, in_dim=feature_dim, p=0.2)
        self.kan_hs_gal2 = KAN_HtrgGraphAttentionLayer(in_dim=feature_dim, out_dim=feature_dim)

    def forward(self, ht1, hs1, S1):
        ht2, hs2, S2 = self.kan_hs_gal1(ht1, hs1, S1)
        ht2_pooled = self.temporal_pool(ht2)
        hs2_pooled = self.spatial_pool(hs2)
        ht3, hs3, S3 = self.kan_hs_gal2(ht2_pooled, hs2_pooled, S2)
        return ht2, hs2, S2, ht3, hs3, S3 # Returning intermediate steps for stacking

class AASIST3_Complete_Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        feature_dim = 64
        
        # 1. Front-End Setup (SincConv + Eq 32 Pre-Encoder)
        self.frontend = AASIST3FrontEndClosed(out_channels=80, kernel_size=251)
        
        # Graph projections mapping
        self.temporal_proj = nn.Linear(6460, feature_dim) 
        self.spatial_proj = nn.Linear(80, feature_dim)
        
        # Learnable Stack Node (S1)
        self.S1 = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # 4 Parallel branches
        self.branches = nn.ModuleList([
            AASIST3_Branch_Block(feature_dim=feature_dim) for _ in range(4)
        ])
        
        # Dropouts definitions
        self.drop_branch = nn.Dropout(p=0.2)
        self.drop_readout = nn.Dropout(p=0.5)
        
        # ARTICLE UPDATE: Final output classifier layer implemented via KAN
        # 5 structural vectors concatenated (Max_t, Mean_t, Max_s, Mean_s, Max_S)
        self.kan_classifier = KAN([5 * feature_dim, 2]) 

    def forward(self, x):
        # x shape: [Batch, Time_Steps=64600]
        
        # Front-end execution (SincConv + MaxPool + BatchNorm + SELU)
        x_hat = self.frontend(x) # [Batch, 80, 6460]
        
        # Base graph axis allocation splits
        ht1 = self.temporal_proj(x_hat)                 # [Batch, 80, feature_dim]
        hs1 = self.spatial_proj(x_hat.transpose(1, 2))  # [Batch, 6460, feature_dim]
        
        batch_size = x.size(0)
        S1 = self.S1.expand(batch_size, -1, -1)         # [Batch, 1, feature_dim]
        
        # Accumulator tensors for summation stacking (Eq 37, 38, 39)
        Ht = torch.zeros_like(ht1)
        Hs = torch.zeros_like(hs1)
        Sf = torch.zeros_like(S1)
        
        # Running 4 branches and summing intermediate outputs directly
        for branch in self.branches:
            ht2, hs2, S2, ht3, hs3, S3 = branch(ht1, hs1, S1)
            
            # ARTICLE EQ 37, 38, 39 Matrix Summation: H = h1 + h2 + h3
            Ht = Ht + ht1 + ht2 + ht3
            Hs = Hs + hs1 + hs2 + hs3
            Sf = Sf + S1 + S2 + S3
            
        # Apply Dropout with probability 0.2 after 4 branches
        Ht = self.drop_branch(Ht)
        Hs = self.drop_branch(Hs)
        Sf = self.drop_branch(Sf)
        
        # STATISTICAL POOLING BLOCK: Extract Max and Mean values
        Hmax_t, _ = torch.max(torch.abs(Ht), dim=1)   # Node-wise max temporal
        Hmean_t   = torch.mean(Ht, dim=1)             # Node-wise mean temporal
        
        Hmax_s, _ = torch.max(torch.abs(Hs), dim=1)   # Node-wise max spatial
        Hmean_s   = torch.mean(Hs, dim=1)             # Node-wise mean spatial
        
        Smax_f, _ = torch.max(torch.abs(Sf), dim=1)   # Maximum Stack Node
        
        # Apply Dropout with probability 0.5 before concatenation
        Hmax_t  = self.drop_readout(Hmax_t)
        Hmean_t = self.drop_readout(Hmean_t)
        Hmax_s  = self.drop_readout(Hmax_s)
        Hmean_s = self.drop_readout(Hmean_s)
        Smax_f  = self.drop_readout(Smax_f)
        
        # FINAL CONCATENATION: Vector L formation
        L = torch.cat([Hmax_t, Hmean_t, Hmax_s, Hmean_s, Smax_f], dim=1)
        
        # Final KAN layer returns predictions logits [Real, Fake]
        logits = self.kan_classifier(L)
        
        return logits
