import torch
import torch.nn as nn
import torch.nn.init as init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import os
import numpy as np
from .wtconv2d import *
import torchvision
from ..modules.block import *
from ..modules.conv import *
from .MambaOut import GatedCNNBlock_BCHW

import torch.nn.functional as F

__all__ = ['PatchEmbed',
           'WTConv2d',
           'C2f_WTConv', 'HWD', 'C2f_MambaOut','C2f_EA','C2f_EA_Lite','C2f_EViT_Lite','C2f_RVB']

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse_self(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

######################################## RepViT start ########################################

class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, use_se=True):
        super(RepViTBlock, self).__init__()

        self.identity = inp == oup
        hidden_dim = 2 * inp

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

class RepViTBlock_EMA(RepViTBlock):
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)
        
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),
        )



class C2f_RVB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock(self.c, self.c, False) for _ in range(n))
######################################## CVPR2025 MambaOut start ########################################

class C2f_MambaOut(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(GatedCNNBlock_BCHW(self.c) for _ in range(n))

######################################## CVPR2025 MambaOut end ########################################

####  external



class C2f_EA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ExternalAttention(self.c) for _ in range(n))

class C2f_EA_Lite(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=4, S=32):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ExternalAttentionLite(self.c, reduction, S) for _ in range(n))

class C2f_EViT_Lite(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_size=5, reduction=8, chunk_ratio=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EfficientViTLiteBlock(self.c, kernel_size, reduction, chunk_ratio) for _ in range(n))

class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Conv1d(d_model,S,1,bias=False)
        self.mv=nn.Conv1d(S,d_model,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()
        self.conv1 = Conv(d_model, d_model, 1, 1)
        self.conv2 = Conv(d_model, d_model, 1, 1)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b,c,h,w = x.shape
        queries = x.view(b,c,h*w)
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model
        out = out.view(b, c, h, w)
        out = self.conv2(out)
        #out = F.relu(out)
        out = out + idn


        return out

class ExternalAttentionLite(nn.Module):

    def __init__(self, d_model, reduction=4, S=32):
        super().__init__()
        d_hidden = max(16, d_model // reduction)
        self.conv1 = Conv(d_model, d_hidden, 1, 1)
        self.mk = nn.Conv1d(d_hidden, S, 1, bias=False)
        self.mv = nn.Conv1d(S, d_hidden, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv2 = Conv(d_hidden, d_model, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        b, c, h, w = x.shape
        attn = self.mk(x.view(b, c, h * w))
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn).view(b, c, h, w)
        out = self.conv2(out)
        return out + identity

class EfficientViTLiteBlock(nn.Module):

    def __init__(self, dim, kernel_size=5, reduction=8, chunk_ratio=0.5):
        super().__init__()
        kernel_size = int(kernel_size)
        gate_channels = max(16, dim // reduction)
        mixed_channels = max(1, int(dim * chunk_ratio))
        self.mixed_channels = mixed_channels
        self.local_mixer = nn.Sequential(
            nn.Conv2d(mixed_channels, mixed_channels, kernel_size, 1, kernel_size // 2, groups=mixed_channels, bias=False),
            nn.BatchNorm2d(mixed_channels),
            nn.SiLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, gate_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, dim, 1, bias=True),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        identity = x
        x_mix, x_keep = torch.split(x, [self.mixed_channels, x.shape[1] - self.mixed_channels], dim=1)
        x_mix = self.local_mixer(x_mix)
        x = torch.cat((x_mix, x_keep), dim=1)
        x = x * self.channel_gate(x)
        return x + identity

######################################## HWD start ########################################

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)
         
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.conv(x)

        return x

######################################## HWD end ########################################

class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x

class Bottleneck_WTConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = WTConv2d(c1, c2)
        self.cv2 = WTConv2d(c2, c2)

class C2f_WTConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_WTConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x
