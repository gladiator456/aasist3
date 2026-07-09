import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AASIST3FrontEndClosed(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels=80, kernel_size=251, sample_rate=16000, stride=1, padding=0):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding

        # SincConv rule: Kernel size odd hona chahiye symmetrical windowing ke liye
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        # Mel-scale filter bank initialization setup
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        # ARTICLE UPDATE: Frequencies ko trainable Parameters (nn.Parameter) banaya
        # Taaki backpropagation me model cutoff frequencies khud seekh sake
        self.mel_low = nn.Parameter(torch.from_numpy(filbandwidthsf[:-1]).float())
        self.mel_band = nn.Parameter(torch.from_numpy(np.diff(filbandwidthsf)).float())

        # Mathematical grids ko register_buffer kiya (non-trainable tensors)
        hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.register_buffer("hsupp", hsupp.float())
        
        window = torch.from_numpy(np.hamming(self.kernel_size).astype(np.float32))
        self.register_buffer("window", window)

        # EQUATION 32 IMPLEMENTATION: Pre-encoder layers setup
        # MaxPool -> BatchNorm -> SELU 
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.selu = nn.SELU()

    def forward(self, x):
        # Input standardisation: [Batch, Time] -> [Batch, Channel=1, Time]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Frequency tracking clamps boundary limits
        low = torch.clamp(self.mel_low, min=0, max=self.sample_rate / 2)
        band = torch.clamp(self.mel_band, min=0, max=self.sample_rate / 2 - low)
        high = low + band

        # Vectorized or clean looping filter generation
        filters = []
        for i in range(self.out_channels):
            fmin = low[i]
            fmax = high[i]
            
            # Sinc wave construction formula
            hHigh = (2 * fmax / self.sample_rate) * torch.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * torch.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            
            filters.append(hideal * self.window)

        # Filters shape: [Out_Channels, 1, Kernel_Size]
        filters = torch.stack(filters).unsqueeze(1)

        # Base feature extraction via Conv1d
        x_features = F.conv1d(x, filters, stride=self.stride, padding=self.padding)

        # ARTICLE EQUATION 32 PIPELINE EXECUTION:
        # \hat{x} = Encoder(SELU(BatchNorm(MaxPool(x))))
        x_pooled = self.max_pool(x_features)
        x_norm = self.batch_norm(x_pooled)
        x_hat = self.selu(x_norm)

        return x_hat
