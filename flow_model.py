import torch
from torch import nn
import math
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))

def flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.MODEL.FLOW.COUPLING_LAYERS):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.MODEL.FLOW.CLAMP_ALPHA,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder

def conditional_flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.MODEL.FLOW.COUPLING_LAYERS):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.MODEL.FLOW.POS_EMBED_DIM,), subnet_constructor=subnet_fc, affine_clamping=args.MODEL.FLOW.CLAMP_ALPHA,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder

def load_flow_model(args, in_channels):
    if args.MODEL.FLOW.FLOW_ARCH == 'flow_model':
        model = flow_model(args, in_channels)
    elif args.MODEL.FLOW.FLOW_ARCH == 'conditional_flow_model':
        model = conditional_flow_model(args, in_channels)
    else:
        raise NotImplementedError('{} is not supported Normalizing Flow!'.format(args.MODEL.FLOW.FLOW_ARCH))
    
    return model


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def load_encoder_arch(model, L):
    cnt = 0
    dims = list()
    layers = ['layers'+str(3+i) for i in range(L)] 
    
    if L >= 2:
        model.encoder.layers[1].blocks[-1].register_forward_hook(get_activation(layers[cnt]))
        dims.append(model.encoder.layers[1].dim)
        cnt = cnt + 1
    if L >= 1:
        model.layers[2].blocks[-1].register_forward_hook(get_activation(layers[cnt]))
        dims.append(model.layers[2].dim)

        cnt = cnt + 1
    return model, layers, dims


def load_encoder_arch_0(model, L):
    cnt = 0
    dims = list()
    layers = ['layers'+str(3+i) for i in range(L)] 
    
    if L >= 3:
        model.encoder.layers[1].blocks[-1].register_forward_hook(get_activation(layers[cnt]))
        dims.append(model.encoder.layers[1].dim)
        cnt = cnt + 1
    if L >= 2:
        model.encoder.layers[2].blocks[-1].register_forward_hook(get_activation(layers[cnt]))
        dims.append(model.encoder.layers[2].dim)
        cnt = cnt + 1
    if L >= 1:
        model.encoder.layers[3].blocks[-1].register_forward_hook(get_activation(layers[cnt]))
        dims.append(model.encoder.layers[3].dim)
        cnt = cnt + 1
    return model.encoder, layers, dims
