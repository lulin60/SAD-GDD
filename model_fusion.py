import torch
import torch.nn as nn
from einops import rearrange

from models.swin_transformer import SwinTransformer
from models.dct import FAD_Head

class Cross_At(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Cross_At, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.q = nn.Linear(dim, dim, bias=True)


    def forward(self, x, y): 
        b, hw, c = x.shape

        q = self.q(x)
        kv = self.kv(y).reshape(b, hw, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v) 
        out = out.permute(0,2,1,3) 
        out = rearrange(out, 'b hw head c1 -> b hw (head c1)', head=self.num_heads, c1 = c // self.num_heads)
        return out


class Two_Stream(nn.Module):
    def __init__(self):
        super().__init__()
        self.dct = FAD_Head(224)
        self.swin_rgb = SwinTransformer()
        self.swin_freh = SwinTransformer()
        self.cross_attention = Cross_At()
    
    def features(self,x):
        rgb = self.swin_rgb.head_features()
        fre = self.dct(x)
        fre = self.swin_freh.head_features()

        for i in range(3): 
            rgb = self.swin_rgb.layers[i](rgb)
            fre = self.swin_freh.layers[i](fre)
            rgb = rgb  + self.cross_attention(rgb ,fre)

        return rgb, fre

    
    def forward(self,x):
        rgb,fre = self.features(x)
        return rgb, fre




