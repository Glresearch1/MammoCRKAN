# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Module that define the core logic of GMIC
"""
import torch
import torch.nn as nn
import numpy as np
# from src.utilities import tools
# import src.modeling.modules as m
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/media/volume/name1/home/exouser/medkaformer/dataset_ddsm/GMIC/src')))
from utilities import tools
# import modeling.modules as m
import modeling.modules_xr as m

from modeling.modules_xr import _init_global  # Initialize _args if necessary
from modeling.taylor_kan import TaylorKAN
# from ball_feature import *

class GMIC(nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()
        _init_global()

        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]

        # construct networks
        # global network
        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)

        # detection network
        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()

        # fusion branch
        self.fusion_dnn = nn.Linear(parameters["post_processing_dim"]+512, parameters["num_classes"])
        self.dropout = nn.Dropout(0.1)

        # self.ofu = m.ball_extract(self.experiment_parameters, self)
        self.ofu = m.OFU(in_channels=parameters["ofu_in_channels"],
            out_channels=parameters["ofu_out_channels"],
            scale=parameters.get("ofu_scale", 2),
            ofu_grid=parameters.get("ofu_grid", "geo"),
            norm=parameters.get("ofu_norm", None),
            act=parameters.get("ofu_act", "gelu"))

    # 创建 TaylorKAN 模型
        self.kan = TaylorKAN(
            layers_hidden=[64, 128, 256, 128, 64, 32],
            order=5,  # 设置泰勒级数的阶数
            scale_base=1.0,
            scale_taylor=1.0,
            base_activation=torch.nn.SiLU,  # 使用 SiLU 作为激活函数
            use_bias=True,
        )



    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            device = torch.device("cuda:{}".format(self.experiment_parameters["gpu_number"]))
            output = output.cuda().to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch(x_original_pytorch[i, 0, :, :],
                                                    self.experiment_parameters["crop_shape"],
                                                    crop_positions[i,j,:],
                                                    output[i,j,:,:],
                                                    method=crop_method)
        return output


    def forward(self, img):
        x = img['image']
        p = 0.5
        # x = img

		# batch_size,num_view,C,H,W = x.shape
		# x = x.reshape(-1, C, H, W)

		# u = self.backbone(x)
		# _,c,h,w = u.shape

		# u = u.reshape(batch_size,num_view,c,h,w)
		# u_m = u[:,0]
		# u_a = u[:,1]

        x_original = x[:,0]
		
        u_a = x[:,1]

        # x_original = self.ofu(x_original) # 学习球面特征
        # u_a = self.ofu(u_a)

        h_g_a, self.saliency_map_a = self.global_network.forward(u_a)
        small_x_locations_a = self.retrieve_roi_crops.forward(u_a, self.cam_size, self.saliency_map_a)
        self.patch_locations_a = self._convert_crop_position(small_x_locations_a, self.cam_size, u_a)


        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # mlo和cc 一起球面，（对齐，对齐，再相加）
        # h_g = self.ofu_2view(h_g, h_g_a)   wrong
        
        # mlo\cc 分别球面，再相加
        h_g = self.dropout(self.ofu(h_g)) # 学习球面特征
        h_g_a = self.dropout(self.ofu(h_g_a))
        h_g += h_g_a
        # print('h_g2',h_g.shape)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        crops_variable_a = self._retrieve_crop(u_a, self.patch_locations_a, self.retrieve_roi_crops.crop_method)
        self.patches_a = crops_variable_a.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)

        batch_size, num_crops, I, J = crops_variable_a.size()
        crops_variable_a = crops_variable_a.view(batch_size * num_crops, I, J).unsqueeze(1)
        # crops_variable_fusion = torch.cat([crops_variable, crops_variable_a], dim=1)
        # print('crops_variable_a shape', crops_variable_a.shape)
        # crops_variable_a = self.kan(crops_variable_a)

        crops_variable_fusion = crops_variable + crops_variable_a

        # h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)  # 原
        h_crops = self.dropout(self.local_network.forward(crops_variable_fusion).view(batch_size, num_crops, -1))

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        z = nn.Dropout(0.3)(z)

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)
        concat_vec = nn.Dropout(p=p)(concat_vec)
        # self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))
        # self.y_fusion = torch.softmax(self.fusion_dnn(concat_vec), dim=1)
        self.y_fusion = self.fusion_dnn(concat_vec)


        return self.y_fusion, self.y_global, self.y_local
        # return self.y_fusion, self.y_global, self.y_local, concat_vec
    



# # 假设参数
# parameters = {
#     "cam_size": (46, 30),  # 模型中定义的CAM尺寸
#     "crop_shape": (256, 256),  # 裁剪尺寸
#     "num_classes": 2,  # 类别数
#     "post_processing_dim": 256,  # 后处理维度
#     "device_type": "gpu",  # 使用CPU还是GPU
#     "K": 6,
#     "gpu_number": 0  # 如果使用GPU, 指定GPU编号
# }

# # 初始化GMIC模型
# gmic_model = GMIC(parameters).cuda()

# x_original = torch.randn(1, 1, 2944, 1920) # single view
# x_original = torch.randn(1, 2, 1, 2944, 1920) # double view

# # 如果模型设定为使用GPU
# if parameters["device_type"] == "gpu":
#     x_original = x_original.cuda()

# # 调用模型的前向传播方法
# output = gmic_model(x_original)

# # 打印输出结果
# print("Model output:", output)