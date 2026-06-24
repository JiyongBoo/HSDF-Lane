"""
hsdflane_fpn.py
"""

import torch
import torchvision as tv
from torch import nn
import torch.nn.functional as F

from .hsdflane import (
    HSDFLane,
    FFN, EncoderLayer, naive_init_module, Residual,
    InstanceEmbedding_offset_y_z, InstanceEmbedding,
    LaneHeadResidual_Instance_with_offset_z, LaneHeadResidual_Instance,
    gradient_scale, FourierPositionalEncoding,
)


# ═══════════════════════════════════════════════════════════════
#  FPN
# ═══════════════════════════════════════════════════════════════
class FPN(nn.Module):
    """C3(512), C4(1024), C5(2048) -> P3(256), P4(256), P5(256)"""
    def __init__(self, in_channels_list=[512, 1024, 2048], out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x_list):
        last_inner = self.inner_blocks[-1](x_list[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for feature, inner_block, layer_block in zip(
            x_list[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode="nearest")
            last_inner = inner_block(feature) + inner_top_down
            results.insert(0, layer_block(last_inner))
        return tuple(results)


# ═══════════════════════════════════════════════════════════════
#  Implicit SDF module (FPN variant — no feature projection)
# ═══════════════════════════════════════════════════════════════
class OptimizedImplicitSDF(nn.Module):
    def __init__(self, feature_dim=256, coord_dim=3, num_freqs=6, hidden_dim=256):
        super().__init__()
        self.pos_encoder = FourierPositionalEncoding(num_freqs=num_freqs)
        encoded_coord_dim = coord_dim + coord_dim * 2 * num_freqs  # 39

        fusion_dim = feature_dim + encoded_coord_dim
        self.layer1 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, hidden_dim),  # skip connection
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, coords):
        encoded_coords = self.pos_encoder(coords)
        fused_input = torch.cat([features, encoded_coords], dim=-1)
        x = self.layer1(fused_input)
        x = torch.cat([x, fused_input], dim=-1)
        return self.layer2(x)


# ═══════════════════════════════════════════════════════════════
#  Main model: HSDFLane_FPN
# ═══════════════════════════════════════════════════════════════
class HSDFLane_FPN(HSDFLane):
    def __init__(self, bev_shape, image_shape, output_2d_shape, train=True,
                 min_spacing=1.5, base_tau=0.3, min_angle=-5.0, max_angle=5.0,
                 num_att=3, fpn_dim=256, use_c5=True, num_points=4):
        # Set before super().__init__() so _init_backbone can access them
        self.use_c5 = use_c5
        self.fpn_dim = fpn_dim
        num_levels = 3 if use_c5 else 2
        super().__init__(bev_shape, image_shape, output_2d_shape, train,
                         min_spacing, base_tau, min_angle, max_angle,
                         feat_dim=fpn_dim, num_att=num_att,
                         num_levels=num_levels, num_points=num_points,
                         transformer_ff_dim=fpn_dim * 2)

    def _init_backbone(self):
        resnet = tv.models.resnet50(pretrained=True)
        self.stem = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1  # C2: 256ch stride-4
        self.layer2 = resnet.layer2  # C3: 512ch stride-8
        self.layer3 = resnet.layer3  # C4: 1024ch stride-16
        if self.use_c5:
            self.layer4 = resnet.layer4  # C5: 2048ch stride-32
        in_ch = [512, 1024, 2048] if self.use_c5 else [512, 1024]
        self.fpn = FPN(in_channels_list=in_ch, out_channels=self.fpn_dim)

    def _init_heads(self, feat_dim, bev_shape, output_2d_shape):
        # Override to substitute the FPN-variant OptimizedImplicitSDF
        super()._init_heads(feat_dim, bev_shape, output_2d_shape)
        self.implicit_mlp = OptimizedImplicitSDF(feature_dim=feat_dim, coord_dim=3, num_freqs=6)

    def _extract_features(self, img):
        x = self.stem(img)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        if self.use_c5:
            features = self.fpn([c3, c4, self.layer4(c4)])
        else:
            features = self.fpn([c3, c4])
        sdf_feat = features[0]
        return sdf_feat, list(features), sdf_feat.shape[2], sdf_feat.shape[3]

    def _build_transformer_src(self, features, device):
        src_list, shapes = [], []
        for feat in features:
            shapes.append((feat.shape[2], feat.shape[3]))
            src_list.append(feat.flatten(2).permute(0, 2, 1))
        src = torch.cat(src_list, dim=1)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return src, spatial_shapes, level_start_index
