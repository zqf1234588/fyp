import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_H, num_W, Ws, Ws, C)
    windows = x.view(-1, window_size * window_size, C)  # (num_windows * B, Ws*Ws, C)
    return windows

def window_reverse(windows, window_size, H, W, C):
    # windows: (num_windows*B, Ws*Ws, C)
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)

class WindowCrossAttentionFusion(nn.Module):
    def __init__(self, dim, window_size=16, heads=4, dropout=0.1):
        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
       
        self.win_dic = {512:16,256:16,128:8,64:8,32:4,16:4}
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape
        
        self.window_size = self.win_dic[H]
       
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        feat_a = F.pad(feat_a, (0, pad_w, 0, pad_h), mode='reflect')
        feat_b = F.pad(feat_b, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, Hp, Wp = feat_a.shape

        
        q = window_partition(feat_a, self.window_size)  # (B*n_win, Ws*Ws, C)
        k = window_partition(feat_b, self.window_size)
        v = window_partition(feat_b, self.window_size)

        # Linear projections
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Multi-head reshape
        B_ = q.shape[0]
        N = q.shape[1]
        q = q.view(B_, N, self.heads, self.head_dim).transpose(1, 2)  # (B_, heads, N, head_dim)
        k = k.view(B_, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B_, N, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B_, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B_, N, self.dim)
        out = self.to_out(out)  # (B_, N, C)

        # Merge windows back
        out = window_reverse(out, self.window_size, Hp, Wp, C)

        # Crop to original size
        out = out[:, :, :H, :W]
        return out, attn
