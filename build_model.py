from .swin_transformer import SwinTransformer
from .dct import FAD_Head
import torch
import torch.nn as nn
from einops import rearrange

def build_encoder(config):
    encoder_type = config.MODEL.ENCODER

    if encoder_type == 'swin':
        enc = SwinTransformer(
            img_size=config.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=0,
            drop_path_rate=config.MODEL.SWIN.ONLINE_DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f'--> Unknown encoder_type: {encoder_type}')

    return enc


class Projector_MLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256,num_layers=2,norm=True):
        super(Projector_MLP, self).__init__()
        self.out_dim = out_dim
        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            if norm==True:
                linear_hidden.append(nn.LayerNorm(inner_dim))

            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim) 


    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x


class Classifier_Mix(nn.Module):
    def __init__(self, in_dim=512, inner_dim=512, out_dim=512,norm=True):
        super(Classifier_Mix, self).__init__()
        
        self.linear_projector = nn.Sequential(
                    nn.Linear(in_dim, inner_dim),
                    nn.LayerNorm(inner_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(inner_dim, out_dim) 
        )
        
        self.linear_projector_mix = nn.Sequential(
                    nn.Linear(196, 196),
                    nn.LayerNorm(196),
                    nn.ReLU(inplace=True),
                    nn.Linear(196, 1) 
        )
        self.head = nn.Linear(512, 2)


    def forward(self, x, w=None):
        x = self.linear_projector(x)

        x = self.linear_projector_mix(x.permute(0,2,1)).squeeze(2)
        x = self.head(x)
        return x


class Cross_At(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Cross_At, self).__init__()
        self.num_heads = num_heads

        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.q = nn.Linear(dim, dim, bias=True)

        self.emb = nn.Linear(dim, dim, bias=True)
        

    def forward(self, x, y): 
        b, hw, c = x.shape

        q = self.q(x)
        q = q.reshape(b, hw, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(y) 
        kv = self.kv(y).reshape(b, hw, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  

        #q = q * self.scale
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1) #[20, 8, 784, 32]

        #attn = (q @ k.transpose(-2, -1))* self.temperature #[20, 8, 784, 784]
        attn = (q @ k.transpose(-2, -1)) #[20, 8, 784, 784]

        attn = attn.softmax(dim=-1)
        out = (attn @ v) #[b head hw c1] [20, 8, 784, 32])

        out = out.permute(0,2,1,3) #[b hw head c1]
        out = rearrange(out, 'b hw head c1 -> b hw (head c1)', head=self.num_heads, c1 = c // self.num_heads) #[20, 784, 256]
        out = self.emb(out)
        return out


class Two_Stream(nn.Module):
    def __init__(self, encoder, cross_at = True, rgb = True, fre = True, fre_flag = 'high'):
        super().__init__()
        self.dct = FAD_Head(224)
        self.swin_rgb = None
        self.swin_fre = None
        self.fre_flag = fre_flag

        if rgb:
            self.swin_rgb = encoder
            self.norm_rgb = nn.LayerNorm(512)

        if fre:
            self.swin_fre = encoder
            self.norm_fre = nn.LayerNorm(512)
        if rgb== True and fre == True:
            self.emb = nn.Linear(512*2, 512, bias=True)

        self.cross_at = cross_at
        if cross_at == True:
            self.cross_attention = nn.ModuleList([Cross_At(256), Cross_At(512), Cross_At(512)])

    def features(self,x):
        out = None
        rgb = torch.zeros(x.shape[0], 196, 512).to(x.device)

        fre = torch.zeros(x.shape[0], 196, 512).to(x.device)
        if self.swin_rgb:
            rgb = self.swin_rgb.head_features(x) 
        if self.swin_fre:
            fre = self.dct(x, fre_flag =  self.fre_flag)
            fre = self.swin_fre.head_features(fre)

        
        for i in range(3): 
            if self.swin_rgb:
                rgb = self.swin_rgb.layers[i](rgb)
            if self.swin_fre:
                fre = self.swin_fre.layers[i](fre)
            if self.cross_at:
                rgb = rgb  + self.cross_attention[i](rgb ,fre)
        if self.swin_rgb:
            rgb = self.norm_rgb(rgb)
            out = rgb
        if self.swin_fre:
            fre = self.norm_fre(fre)
            out = fre

        if self.swin_rgb and self.swin_fre:
            out = torch.cat((rgb,fre),dim=-1)
            out = self.emb(out)

           
        return out, rgb, fre
    

    def forward(self,x):
        out = self.features(x)
        return out

