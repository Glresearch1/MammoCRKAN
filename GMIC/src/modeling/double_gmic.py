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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC. If not, see <http://www.gnu.org/licenses/>.

"""Double-view GMIC model with a KAN fusion classifier."""

import numpy as np
import torch
import torch.nn as nn

from models.KANLinear import KAN

from ..utilities import tools
from . import modules as m


class GMIC(nn.Module):
    """GMIC model for paired mammography views."""

    def __init__(self, parameters):
        super().__init__()
        m._init_global()

        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]
        self.store_patches = parameters.get("store_patches", False)

        self.global_network = m.GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters, self)
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters, self)

        self.local_network = m.LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        self.attention_module = m.AttentionModule(self.experiment_parameters, self)
        self.attention_module.add_layers()

        fusion_in_features = parameters["post_processing_dim"] + 512
        self.fusion_dnn = KAN(
            layers_hidden=[fusion_in_features, parameters["num_classes"]],
            grid_size=parameters.get("kan_grid_size", 5),
            spline_order=parameters.get("kan_spline_order", 3),
            scale_noise=parameters.get("kan_scale_noise", 0.1),
            scale_base=parameters.get("kan_scale_base", 1.0),
            scale_spline=parameters.get("kan_scale_spline", 1.0),
        )

        self.feature_dropout = nn.Dropout(parameters.get("feature_dropout", 0.0))
        self.fusion_dropout = nn.Dropout(parameters.get("fusion_dropout", 0.3))

        self.ofu = m.OFU(
            in_channels=parameters["ofu_in_channels"],
            out_channels=parameters["ofu_out_channels"],
            scale=parameters.get("ofu_scale", 2),
            ofu_grid=parameters.get("ofu_grid", "geo"),
            norm=parameters.get("ofu_norm", None),
            act=parameters.get("ofu_act", "gelu"),
        )

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """Convert crop locations from CAM coordinates to image coordinates."""

        h, w = cam_size
        _, _, image_h, image_w = x_original.size()

        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w

        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"

        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * image_h), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * image_w), -1)
        return np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """Crop high-resolution image regions selected from the CAM."""

        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]
        output = x_original_pytorch.new_empty((batch_size, num_crops, crop_h, crop_w))

        for batch_idx in range(batch_size):
            for crop_idx in range(num_crops):
                tools.crop_pytorch(
                    x_original_pytorch[batch_idx, 0, :, :],
                    self.experiment_parameters["crop_shape"],
                    crop_positions[batch_idx, crop_idx, :],
                    output[batch_idx, crop_idx, :, :],
                    method=crop_method,
                )
        return output

    def _extract_view(self, view):
        feature_map, saliency_map = self.global_network.forward(view)
        cam_size = saliency_map.shape[-2:]
        small_locations = self.retrieve_roi_crops.forward(view, cam_size, saliency_map)
        patch_locations = self._convert_crop_position(small_locations, cam_size, view)
        crops = self._retrieve_crop(view, patch_locations, self.retrieve_roi_crops.crop_method)
        return feature_map, saliency_map, patch_locations, crops

    def forward(self, batch):
        x = batch["image"]
        if x.dim() != 5 or x.size(1) < 2:
            raise ValueError("Expected batch['image'] with shape [batch, 2, channels, height, width].")

        primary_view = x[:, 0]
        auxiliary_view = x[:, 1]

        h_aux, self.saliency_map_a, self.patch_locations_a, crops_aux = self._extract_view(auxiliary_view)
        h_primary, self.saliency_map, self.patch_locations, crops_primary = self._extract_view(primary_view)

        if self.store_patches:
            self.patches = crops_primary.detach().cpu().numpy()
            self.patches_a = crops_aux.detach().cpu().numpy()

        h_global = self.feature_dropout(self.ofu(h_primary))
        h_global_aux = self.feature_dropout(self.ofu(h_aux))
        h_global = h_global + h_global_aux

        global_saliency = 0.5 * (self.saliency_map + self.saliency_map_a)
        self.y_global = self.aggregation_function.forward(global_saliency)

        batch_size, num_crops, crop_h, crop_w = crops_primary.size()
        crops_primary = crops_primary.view(batch_size * num_crops, crop_h, crop_w).unsqueeze(1)
        crops_aux = crops_aux.view(batch_size * num_crops, crop_h, crop_w).unsqueeze(1)
        crops_fusion = crops_primary + crops_aux

        h_crops = self.local_network.forward(crops_fusion).view(batch_size, num_crops, -1)
        h_crops = self.feature_dropout(h_crops)

        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        global_vec = torch.amax(h_global, dim=(2, 3))
        concat_vec = torch.cat([global_vec, z], dim=1)
        concat_vec = self.fusion_dropout(concat_vec)
        self.y_fusion = self.fusion_dnn(concat_vec)

        return self.y_fusion, self.y_global, self.y_local
