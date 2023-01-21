import torch
from torch import nn
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.time_dim = time_dim
        self.in_c = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(in_channels=256, out_channels=256)
        self.sa3 = SelfAttention(256, 8)
        
        self.bt1 = DoubleConv(in_channels=256, out_channels=512)
        self.bt2 = DoubleConv(in_channels=512, out_channels=512)
        self.bt3 = DoubleConv(in_channels=512, out_channels=256)
        
        self.up1 = Up(in_channels=512, out_channels=128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(in_channels=256, out_channels=64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(in_channels=128, out_channels=64)
        self.sa6 = SelfAttention(64, 64)
        
        self.out_c = nn.Conv2d(64, out_channels=out_channels, kernel_size=1)
        
    def position_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        
        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
        
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        
        return pos_enc
    
    def forward(self, x, t):
        t = t.to(self.device)
        x = x.to(self.device)
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.position_encoding(t, self.time_dim)
        
        x1 = self.in_c(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        x4 = self.bt1(x4)
        x4 = self.bt2(x4)
        x4 = self.bt3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.out_c(x)
        
        return output
         
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        
        self.residual = residual
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels) 
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        
        else:
            return self.double_conv(x)
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emd_dim=256):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emd_dim, out_features=out_channels)
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        return x + emb

   
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emd_dim=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emd_dim, out_features=out_channels)
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        
        self.channels = channels
        self.size = size
        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)