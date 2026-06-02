import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from packaging.version import Version

"""TPAMI2023

1. 球面几何特征的生成
首先，在 OFU 模块中，我们通过一个函数 uv_grid 来生成球面坐标。这一步非常关键，因为我们处理的是全景图像，而全景图像本身在球面坐标系下有特定的几何特性。uv_grid 函数通过计算每个像素在球面上的坐标来生成 u 和 v，其中 u 代表水平角度，v 代表垂直角度。这些球面坐标帮助我们将二维图像映射到球面上。

2. 球面几何特征与特征图的结合
接下来，球面几何特征会与输入的特征图结合。具体来说，我们通过 register_embed 函数，将球面坐标经过一系列的 sin 和 cos 运算，生成了一个带有球面几何特征的网格。这个网格会扩展到与输入特征图相同的尺寸，并用于后续的重新采样过程。

3. 偏移场与特征重新采样
然后，我们在 wraped_sample 函数中，将球面几何特征和输入的特征图拼接在一起，形成一个新的输入。通过 ofu 模块，我们可以得到一个偏移场，这个偏移场将帮助我们更好地对输入特征图进行采样。具体来说，偏移场会告诉我们特征图中每个像素点的正确位置，使得我们可以在球面坐标系下进行合理的采样。

最后，我们使用 spherical_grid_sample 函数来完成特征的重新采样。这个函数会根据球面几何特征对输入特征图进行采样，输出一个在球面坐标系下优化过的特征图。这一过程帮助我们在全景图像处理任务中，提升了对图像细节的捕捉，特别是在边缘和深度估计等方面效果显著。
"""

# Initialize the global _args variable
def _init_global():
    global _args
    _args = OrderedDict({
        'model': {
            'padding': 'circpad'  # Default padding value
        }
    })

def spherical_grid_sample(x,grid,clip=False,inplace=False,indexing='ij'):
    B,C,H,W = x.shape
    assert len(grid.shape) == 4 and grid.shape[-1] == 2
    # assert tuple(grid.shape) == (B,H,W,2)

    if clip:
        grid = spherical_reminder(grid,[H,W],inplace=inplace,indexing=indexing)
    if indexing == 'ij':
        grid_h = grid[...,0]
        grid_w = grid[...,1]
    else:
        grid_h = grid[...,1]
        grid_w = grid[...,0]

    x = torch.cat([x,x[...,:1]],dim=-1)
    grid = torch.stack([grid_w/W*2-1,grid_h/(H-1)*2-1],dim=-1)
    y = torch.nn.functional.grid_sample(x,grid,align_corners=True)
    return y

def _make_norm(norm,layers,**kargs):
    if norm is None or norm == 'idt' or norm == 'none':
        return nn.Identity()
    elif norm == 'bn':
        return nn.BatchNorm2d(layers)
    elif norm == 'inst':
        return nn.InstanceNorm2d(layers)
    elif norm == 'gn':
        if not 'groups' in kargs:
            return nn.GroupNorm(32,layers)
        else:
            return nn.GroupNorm(kargs['groups'],layers)
    else:
        raise NotImplementedError

def _make_act(act,**kargs):
    if act is None or act == 'idt':
        return nn.Identity()
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.01,inplace=True)
    elif act == 'orelu':
        return nn.ReLU(inplace=False)
    elif act == 'olrelu':
        return nn.LeakyReLU(negative_slope=0.01,inplace=False)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError

def _set_value(key, value):
    _args[key] = value

def _get_value(key):
    try:
        return _args[key]
    except:
        print('error!')
        exit(-1)

# _make_pad function to handle padding selection based on _args
def _make_pad(padding=0, pad=None, **kargs):
    if pad is None and 'padding' in _args['model']:
        pad = _args['model']['padding']
    if pad == 'circpad':
        return nn.ReflectionPad2d(padding)  # Replace with custom CircPad if necessary
    elif pad == 'lrpad':
        return nn.ReplicationPad2d(padding)  # Replace with actual implementation
    elif pad == 'zeropad':
        return nn.ZeroPad2d(padding)
    else:
        return nn.ReflectionPad2d(padding)  # Default to ReflectionPad2d

# Functions for grid creation
def xy_grid(h, w, dim=0):
    if Version(torch.__version__) >= Version('1.10.0'):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    else:
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    return torch.stack([x.cuda(), y.cuda()], dim=dim)

def uv_grid(h, w, dim=0):
    if Version(torch.__version__) >= Version('1.10.0'):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    else:
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    u = (x.cuda().type(torch.float32) - (w - 1) / 2) / w * (2 * np.pi)
    v = -(y.cuda().type(torch.float32) - (h - 1) / 2) / h * np.pi
    return torch.stack([u, v], dim=dim)

class OFU(nn.Module):
    def __init__(self, in_channels, out_channels=None, scale=2, ofu_grid='geo', norm=None, act='gelu', force=False):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        mid_channels = 32
        self.scale = scale
        self.ofu_grid_type = ofu_grid
        self.ofu_grid = None
        self.xy_grid = None
        self.pad = _make_pad(1)

        grid_channels = 5

        self.of = nn.Sequential(
            _make_pad(1),
            nn.Conv2d(in_channels + grid_channels, mid_channels, kernel_size=3, bias=not norm),
            _make_norm(norm, mid_channels),
            _make_act(act),
            nn.Conv2d(mid_channels, 2, 1)
        )
        if in_channels == out_channels and not force:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def register_embed(self, shape):
        b, _, h, w = shape
        uv = uv_grid(h, w).view(1, 2, h, w)
        grid = torch.cat([torch.sin(uv[:, :1]), torch.cos(uv[:, :1]), torch.sin(uv[:, 1:]), torch.cos(uv[:, 1:]),
                          torch.cos(uv[:, :1]) * torch.cos(uv[:, 1:])], dim=1)
        self.ofu_grid = grid.expand([b, -1, -1, -1])

    def wraped_sample(self, x):
        b, _, h, w = x.shape

        if self.ofu_grid is None or not self.ofu_grid.shape[0] == b or not tuple(self.ofu_grid.shape[-2:]) == tuple(x.shape[-2:]):
            self.register_embed(x.shape)

        if self.xy_grid is None or not tuple(self.xy_grid.shape[-2:]) == tuple(x.shape[-2:]):
            self.xy_grid = xy_grid(h, w).view(1, 2, h, w)

        of_input = torch.cat([x, self.ofu_grid], dim=1)
        of = self.of(of_input)  # [b, 2, h, w]
        of = of + self.xy_grid  # [b, 2, h, w]

        of = of.permute([0, 2, 3, 1])  # [b, h, w, 2]
        y = spherical_grid_sample(x, of, inplace=False, indexing='xy')

        return y

    def forward(self, x):

        original_size = x.size()  # 保存输入的原始大小

        x = self.pad(x)

        #  上采样的操作 
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = x[:, :, self.scale:-self.scale, self.scale:-self.scale]

        y = self.wraped_sample(x)
        y = self.conv(y)

        # # 添加反向插值，将输出缩回到原始大小
        y = F.interpolate(y, size=(original_size[2], original_size[3]), mode='bilinear', align_corners=False)

        return y


# # Main function to test the OFU block
# if __name__ == '__main__':
#     _init_global()  # Initialize _args

#     block = OFU(in_channels=3, out_channels=3).cuda()

#     input_tensor = torch.rand(1, 3, 64, 64).cuda()

#     output_tensor = block(input_tensor)

#     print("Input size:", input_tensor.size())
#     print("Output size:", output_tensor.size())


