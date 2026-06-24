"""
hsdflane.py
"""

import torch
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from .deformable import (
    _get_activation_fn,
    IdentityMSDeformAttn,
    DropoutMSDeformAttn,
    PositionEmbeddingLearned,
)


class FFN(nn.Module):
    def __init__(self,
                 d_model=256,
                 dim_ff=1024,
                 activation='relu',
                 ffn_dropout=0.0,
                 add_identity=True):
        super().__init__()
        self.d_model = d_model
        self.feedforward_channels = dim_ff

        self.linear1 = nn.Linear(d_model, dim_ff)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(ffn_dropout)

        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.add_identity = add_identity
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight.data)
        constant_(self.linear1.bias.data, 0.0)
        xavier_uniform_(self.linear2.weight.data)
        constant_(self.linear2.bias.data, 0.0)

    def forward(self, x, identity=None):
        inter = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        out = self.dropout2(inter)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=None,
                 dim_ff=None,
                 activation='relu',
                 ffn_dropout=0.0,
                 num_levels=4,
                 num_points=8,
                 num_heads=8):
        super().__init__()
        self.fp16_enabled = False

        self.self_attn = IdentityMSDeformAttn(d_model=d_model, n_levels=1)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = DropoutMSDeformAttn(
            d_model=d_model,
            n_levels=num_levels,
            n_points=num_points,
            n_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FFN(
            d_model=d_model,
            dim_ff=dim_ff,
            activation=activation,
            ffn_dropout=ffn_dropout,
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,
                query=None,
                value=None,
                bev_pos=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                spatial_shapes=None,
                level_start_index=None):
        identity = query

        temp_key = temp_value = query
        query = self.self_attn(
            query + bev_pos,
            reference_points=ref_2d,
            input_flatten=temp_value,
            input_spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            input_level_start_index=torch.tensor([0], device=query.device),
            identity=identity,
        )
        identity = query

        query = self.norm1(query)

        query = self.cross_attn(
            query,
            reference_points=ref_3d,
            input_flatten=value,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
        )
        query = query + identity

        query = self.norm2(query)
        query = self.ffn(query)
        query = self.norm3(query)
        return query


def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super().__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super().__init__()
        self.neck_new = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1, bias=True),
        )

        self.m_offset_new = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1, bias=True),
        )

        self.me_new = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, co, 3, 1, 1, bias=True),
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1, bias=True),
        )

        self.me = nn.Sequential(
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, co, 3, 1, 1, bias=True),
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super().__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super().__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),
            nn.Upsample(size=output_size),
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                ),
            ),
        )
        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


# ═══════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════
class _GradientScalerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def gradient_scale(x, scale=0.2):
    return _GradientScalerFn.apply(x, scale)


# ═══════════════════════════════════════════════════════════════
#  Implicit HSDF module
# ═══════════════════════════════════════════════════════════════
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * torch.pi))
            out.append(torch.cos(x * freq * torch.pi))
        return torch.cat(out, dim=-1)


class OptimizedImplicitSDF(nn.Module):
    def __init__(self, feature_dim=1024, coord_dim=3, num_freqs=6, hidden_dim=256):
        super().__init__()
        self.pos_encoder = FourierPositionalEncoding(num_freqs=num_freqs)
        encoded_coord_dim = coord_dim + coord_dim * 2 * num_freqs  # 39

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        fusion_dim = hidden_dim + encoded_coord_dim
        self.layer1 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, hidden_dim),  # Skip Connection
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, coords):
        encoded_coords = self.pos_encoder(coords)
        proj_features = self.feature_proj(features)
        fused_input = torch.cat([proj_features, encoded_coords], dim=-1)
        x = self.layer1(fused_input)
        x = torch.cat([x, fused_input], dim=-1)  # Skip
        return self.layer2(x)


# ═══════════════════════════════════════════════════════════════
#  Main model : HSDF-Lane
# ═══════════════════════════════════════════════════════════════
class HSDFLane(nn.Module):
    def create_slope_heightmap(self, angle, x_forward_map):
        """
        angle: slope angle in degree.
        x_forward_map: (H, W) BEV forward distance map in meters.
        """
        rad = torch.tensor(angle * (torch.pi / 180), dtype=x_forward_map.dtype,
                           device=x_forward_map.device)
        return x_forward_map * torch.tan(rad)

    def __init__(self, bev_shape, image_shape, output_2d_shape, train=True,
                 min_spacing=1.5, base_tau=0.3, min_angle=-5.0, max_angle=5.0,
                 feat_dim=1024, num_att=2, num_levels=1, num_points=4,
                 transformer_ff_dim=None):
        """
        min_spacing:       uniform z-spacing for ray marching (m).
        base_tau:          base softmax temperature. Scaled per-row by Δz/δ.
        min_angle:         slope angle (deg) for z_min_anchor heightmap.
        max_angle:         slope angle (deg) for z_max_anchor heightmap.
        feat_dim:          feature channel dimension (1024 for base, 256 for FPN).
        num_att:           number of transformer encoder layers.
        num_levels:        number of feature scales for MSDA (1 for base, 3 for FPN).
        num_points:        sampling points per head in cross-attention.
        transformer_ff_dim: FFN hidden dim in transformer (defaults to feat_dim).
        """
        super(HSDFLane, self).__init__()
        self.min_spacing = min_spacing
        self.is_train = train

        self._init_backbone()
        self._init_bev_sampling(bev_shape, min_spacing, base_tau, min_angle, max_angle)
        self._init_heads(feat_dim, bev_shape, output_2d_shape)
        self._init_transformer(feat_dim, bev_shape, num_att, num_levels, num_points,
                               transformer_ff_dim if transformer_ff_dim is not None else feat_dim)

    # ────────────────────────────────────────────────────
    #  Init helpers
    # ────────────────────────────────────────────────────
    def _init_backbone(self):
        self.bb = nn.Sequential(*list(tv.models.resnet50(pretrained=True).children())[:-3])

    def _init_bev_sampling(self, bev_shape, min_spacing, base_tau, min_angle, max_angle):
        self.register_buffer('matrix_cameraw2camera', torch.tensor(
            ([[-0., -1., -0., -0.],
              [-0., -0., -1., -0.],
              [ 1.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  1.]]), dtype=torch.float32))

        x = torch.linspace(0, bev_shape[0] - 1, bev_shape[0])
        y = torch.linspace(0, bev_shape[1] - 1, bev_shape[1])
        xv_, yv_ = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('xv_', xv_.clone())
        self.register_buffer('yv_', yv_.clone())

        # Cell-center BEV mapping: physical = center of each 0.5m cell
        xv_phys = 103 - ((self.xv_ + 0.5) / 2)
        yv_phys = (self.yv_ + 0.5) / 2 - 12
        self.register_buffer('z_min_anchor', self.create_slope_heightmap(min_angle, xv_phys))
        self.register_buffer('z_max_anchor', self.create_slope_heightmap(max_angle, xv_phys))

        ones_map = torch.ones_like(self.xv_)
        self.register_buffer('base_coords', torch.stack((yv_phys, xv_phys, ones_map), dim=-1))
        self.register_buffer('x_norm', (xv_phys - 53.0) / 50.0)
        self.register_buffer('y_norm', (yv_phys - 0.0) / 12.0)

        z_range_max = (self.z_max_anchor - self.z_min_anchor).max().item()
        self.max_K = int(torch.ceil(torch.tensor(z_range_max / min_spacing)).item()) + 1
        self.max_K = max(self.max_K, 2)

        k_idx = torch.arange(self.max_K).float().view(self.max_K, 1, 1)  # (max_K, 1, 1)
        z_all_raw = self.z_min_anchor.unsqueeze(0) + k_idx * min_spacing  # (max_K, H, W)
        z_all = torch.min(z_all_raw, self.z_max_anchor.unsqueeze(0))      # (max_K, H, W)
        self.register_buffer('z_all', z_all)

        # Padding mask: True at invalid positions (beyond z_max)
        k_pad_mask = z_all_raw > (self.z_max_anchor.unsqueeze(0) + 1e-4)  # (max_K, H, W)
        k_pad_mask[:2, :, :] = False
        self.register_buffer('k_pad_mask', k_pad_mask.unsqueeze(0))        # (1, max_K, H, W)
        self.register_buffer('_sdf_valid_mask',
                             (~k_pad_mask).float().unsqueeze(0))           # (1, max_K, H, W)

        # Pre-compute 1D indices of valid points; permute to (H, W, max_K) to match feat_all spatial order
        mask_3d = (~k_pad_mask).permute(1, 2, 0)                           # (H, W, max_K)
        valid_indices = mask_3d.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer('valid_indices', valid_indices)
        self.num_valid_points = valid_indices.shape[0]

        # Adaptive Temperature
        dz_map = self.z_max_anchor - self.z_min_anchor                # (H, W)
        tau_scale = (dz_map / min_spacing).clamp(0.3, 3.0)
        self.register_buffer('adaptive_tau',
                             (base_tau * tau_scale).unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)

        # Pre-computed batched tensors (loop-free forward)
        self.register_buffer('base_coords_tiled',
            self.base_coords.unsqueeze(0).expand(self.max_K, -1, -1, -1)
            .contiguous().view(self.max_K * bev_shape[0], bev_shape[1], 3))

        x_exp = self.x_norm.unsqueeze(-1).expand(-1, -1, self.max_K)
        y_exp = self.y_norm.unsqueeze(-1).expand(-1, -1, self.max_K)
        z_range = (self.z_max_anchor - self.z_min_anchor).clamp(min=1e-6).unsqueeze(0)
        z_norm_buf = 2.0 * (z_all - self.z_min_anchor.unsqueeze(0)) / z_range - 1.0
        self.register_buffer('coord_norm_all',
            torch.stack([x_exp, y_exp, z_norm_buf.permute(1, 2, 0)], dim=-1))

        # Pre-flatten valid coords and store as buffer
        coord_flat_all = self.coord_norm_all.reshape(-1, 3)            # (H*W*max_K, 3)
        self.register_buffer('coord_valid',
                             coord_flat_all[valid_indices].contiguous())  # (N_valid, 3)

    def _init_heads(self, feat_dim, bev_shape, output_2d_shape):
        self.bev_smooth = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim, bias=False),
            nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.constant_(self.bev_smooth[1].weight, 0.0)

        self.height_smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.height_smooth.weight, 0.0)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # for LSPE: rank-1 lane semantic vector
        self.lane_semantic_vector = nn.Parameter(torch.empty(1, 1, feat_dim))
        nn.init.normal_(self.lane_semantic_vector, std=0.02)

        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=feat_dim)
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=feat_dim)

        self.implicit_mlp = OptimizedImplicitSDF(feature_dim=feat_dim, coord_dim=3, num_freqs=6)

    def _init_transformer(self, feat_dim, bev_shape, num_att, num_levels, num_points, ff_dim):
        self.num_att = num_att
        self.num_levels = num_levels
        self.num_points = num_points
        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        bev_h, bev_w = bev_shape

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, bev_h - 0.5, bev_h),
            torch.linspace(0.5, bev_w - 0.5, bev_w),
            indexing='ij',
        )
        ref_point = torch.stack(
            (ref_x.reshape(-1)[None] / bev_w,
             ref_y.reshape(-1)[None] / bev_h), -1
        ).repeat(1, 1, 1).unsqueeze(2)
        self.register_buffer('_ref_2d', ref_point)

        self.query_embeds.append(nn.Embedding(bev_h * bev_w, feat_dim))
        self.pe.append(PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=feat_dim // 2))

        for j in range(num_att):
            self.el.append(EncoderLayer(
                d_model=feat_dim, dim_ff=ff_dim,
                num_levels=num_levels, num_points=num_points, num_heads=4,
            ))

    # ────────────────────────────────────────────────────
    #  Feature extraction (overridden by subclasses)
    # ────────────────────────────────────────────────────
    def _extract_features(self, img):
        """Returns (sdf_feat, features_list, img_h, img_w)."""
        feat = self.bb(img)
        return feat, [feat], feat.shape[2], feat.shape[3]

    def _build_transformer_src(self, features, device):
        """Returns (src, spatial_shapes, level_start_index)."""
        src = features[0].flatten(2).permute(0, 2, 1)
        h, w = features[0].shape[2], features[0].shape[3]
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=device)
        level_start_index = torch.zeros((1,), dtype=torch.long, device=device)
        return src, spatial_shapes, level_start_index

    # ────────────────────────────────────────────────────
    #  BEV ↔ Image Projection
    # ────────────────────────────────────────────────────
    def height4featuremap(self, heightmaps, intrinsics, extrinsics, featuremap,
                          input_2d=(800, 600), anchor=False, row_range=None):
        device = intrinsics.device
        batch_size, c1, h1, w1 = featuremap.shape
        if anchor:
            if heightmaps.dim() == 2:
                heightmaps = heightmaps.unsqueeze(0).expand(batch_size, -1, -1)
            elif heightmaps.dim() == 4:
                heightmaps = heightmaps.squeeze(1)
            batch_size, bev_height, bev_width = heightmaps.shape
            z_grid = heightmaps
        else:
            batch_size, _, bev_height, bev_width = heightmaps.shape
            z_grid = heightmaps.squeeze(1)

        extrinsics = self.matrix_cameraw2camera @ extrinsics

        if row_range is not None:
            r0, r1 = row_range
            batch_base = self.base_coords[r0:r1].unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            batch_base = self.base_coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

        bev_coords = torch.empty(
            (batch_size, bev_height, bev_width, 4), device=device, dtype=torch.float32)
        bev_coords[..., 0] = batch_base[..., 0]
        bev_coords[..., 1] = batch_base[..., 1]
        bev_coords[..., 2] = z_grid
        bev_coords[..., 3] = batch_base[..., 2]
        bev_coords = bev_coords.reshape(batch_size, bev_height * bev_width, 4).permute(0, 2, 1)

        bev_points_camera = torch.bmm(extrinsics, bev_coords)
        intrinsic1 = intrinsics.clone()
        intrinsic1[:, 0, :] = intrinsic1[:, 0, :] * w1 / input_2d[0]
        intrinsic1[:, 1, :] = intrinsic1[:, 1, :] * h1 / input_2d[1]

        with torch.cuda.amp.autocast(enabled=False):
            bev_points_image = torch.bmm(
                intrinsic1.float(), bev_points_camera[:, :3, :].float()
            ).permute(0, 2, 1)
        depth = bev_points_image[..., 2:].clamp(min=1e-3)
        bev_points_image = bev_points_image[..., :2] / depth

        if anchor:
            bev_points_image[..., 0] = (bev_points_image[..., 0] / (w1 - 1)) * 2 - 1
            bev_points_image[..., 1] = (bev_points_image[..., 1] / (h1 - 1)) * 2 - 1
            bev_points_image = bev_points_image.reshape(
                batch_size, bev_height, bev_width, 2).float()
            return F.grid_sample(featuremap, bev_points_image, align_corners=True)
        else:
            bev_points_image[..., 0] = (bev_points_image[..., 0] / (w1 - 1))
            bev_points_image[..., 1] = (bev_points_image[..., 1] / (h1 - 1))
            return bev_points_image.float()

    # ────────────────────────────────────────────────────
    #  Forward Pass
    # ────────────────────────────────────────────────────
    def forward(self, img, intrinsic, extrinsic, prev=False):
        sdf_feat, features, img_h, img_w = self._extract_features(img)
        bs, c = sdf_feat.shape[0], sdf_feat.shape[1]
        h_bev, w_bev = 200, 48
        device = img.device

        # ── 1. Batched BEV → Image projection ──
        n_spatial = self.max_K * h_bev
        z_flat = self.z_all.reshape(n_spatial, w_bev)
        z_batch = z_flat.unsqueeze(0).expand(bs, -1, -1)

        extrinsics_cam = self.matrix_cameraw2camera @ extrinsic
        batch_base = self.base_coords_tiled.unsqueeze(0).expand(bs, -1, -1, -1)

        bev_coords = torch.empty(
            (bs, n_spatial, w_bev, 4), device=device, dtype=torch.float32)
        bev_coords[..., 0] = batch_base[..., 0]
        bev_coords[..., 1] = batch_base[..., 1]
        bev_coords[..., 2] = z_batch
        bev_coords[..., 3] = batch_base[..., 2]
        bev_coords = bev_coords.reshape(bs, n_spatial * w_bev, 4).permute(0, 2, 1)

        bev_pts_cam = torch.bmm(extrinsics_cam, bev_coords)
        intrinsic1 = intrinsic.clone()
        intrinsic1[:, 0, :] *= img_w / 800
        intrinsic1[:, 1, :] *= img_h / 600

        with torch.cuda.amp.autocast(enabled=False):
            bev_pts_img = torch.bmm(
                intrinsic1.float(), bev_pts_cam[:, :3, :].float()
            ).permute(0, 2, 1)
        depth = bev_pts_img[..., 2:].clamp(min=1e-3)
        bev_pts_img = bev_pts_img[..., :2] / depth
        bev_pts_img[..., 0] = (bev_pts_img[..., 0] / (img_w - 1)) * 2 - 1
        bev_pts_img[..., 1] = (bev_pts_img[..., 1] / (img_h - 1)) * 2 - 1
        grid = bev_pts_img.reshape(bs, n_spatial, w_bev, 2)

        # ── 2. Single grid_sample ──
        with torch.backends.cudnn.flags(enabled=False):
            feat_all = F.grid_sample(
                sdf_feat.contiguous(), grid.contiguous(), align_corners=True)
        feat_all = feat_all.reshape(
            bs, c, self.max_K, h_bev, w_bev).permute(0, 3, 4, 2, 1)  # (B,H,W,max_K,c)

        # ── 3. Static Indexing → Gather → MLP → Scatter ──
        n_hwk = h_bev * w_bev * self.max_K

        # feat_all is non-contiguous after permute; make contiguous before reshape
        feat_flat = feat_all.contiguous().reshape(bs, n_hwk, c)        # (B, H*W*K, C)

        # Gather valid points using pre-computed indices
        feat_valid = feat_flat[:, self.valid_indices, :]               # (B, N_valid, C)

        # Use pre-extracted valid coord buffer from __init__
        coord_valid = self.coord_valid.unsqueeze(0).expand(bs, -1, -1) # (B, N_valid, 3)

        # Run MLP on valid points only
        sdf_valid = self.implicit_mlp(feat_valid, coord_valid).squeeze(-1)  # (B, N_valid)

        # Scatter valid results into dense tensor (padded with 100.0)
        sdf_dense = torch.full(
            (bs, n_hwk), 100.0, device=device, dtype=sdf_valid.dtype)
        sdf_dense[:, self.valid_indices] = sdf_valid
        sdf_all = sdf_dense.reshape(bs, h_bev, w_bev, self.max_K).permute(0, 3, 1, 2)
        # sdf_all: (B, max_K, H, W) — padded positions are 100.0

        # ── 4. Softmax + sub-bin heightmap rendering ──
        # Padded positions already = 100.0; softmax weight ≈ 0, no masked_fill needed
        with torch.cuda.amp.autocast(enabled=False):
            sdf_f32 = sdf_all.float()

            weights = F.softmax(
                -torch.abs(sdf_f32) / self.adaptive_tau, dim=1)

            z_all_batch = self.z_all.unsqueeze(0).expand(bs, -1, -1, -1)
            z_surface = z_all_batch - sdf_f32
            heightmap = torch.sum(weights * z_surface, dim=1)          # (B,H,W)

            # Heightmap Smoothing
            heightmap_4d = heightmap.unsqueeze(1)
            heightmap_4d = heightmap_4d + self.height_smooth(heightmap_4d)
            heightmap = heightmap_4d.squeeze(1)

            # ── 5. HSDF feature rendering ──
            w_exp = weights.permute(0, 2, 3, 1).unsqueeze(-1)
            rendered_feat = torch.sum(feat_all.float() * w_exp, dim=3)
            aligned_lane_feature = rendered_feat.permute(0, 3, 1, 2)   # (B,c,H,W)

        if prev:
            return heightmap

        # HSDF feature smoothing
        aligned_lane_feature = aligned_lane_feature + self.bev_smooth(aligned_lane_feature)

        # ── 6. Heatmap prediction ──
        heightmap_4d = heightmap.unsqueeze(1)
        aligned_lane_feature = gradient_scale(aligned_lane_feature, scale=0.2)
        lane_heatmap = self.heatmap_head(aligned_lane_feature)

        # ── 7. LSPE (semantic positional encoding) ──
        heatmap_prob = lane_heatmap.detach().flatten(2).permute(0, 2, 1)
        semantic_pos = heatmap_prob * self.lane_semantic_vector

        bev_pos = self.pe[0](heightmap_4d).to(sdf_feat.dtype).flatten(2).permute(0, 2, 1)
        bev_pos = bev_pos + semantic_pos

        # ── 8. Transformer ──
        src, spatial_shapes, level_start_index = self._build_transformer_src(features, device)
        ref_pnts = self.height4featuremap(
            heightmap_4d, intrinsic, extrinsic, sdf_feat,
            input_2d=(800, 600)).unsqueeze(-2).repeat(1, 1, self.num_levels, 1).contiguous()

        query_embed = self.query_embeds[0].weight.unsqueeze(0).repeat(bs, 1, 1)
        ref_2d = self._ref_2d.repeat(bs, 1, 1, 1)

        for j in range(self.num_att):
            query_embed = self.el[j](
                query=query_embed, value=src, bev_pos=bev_pos,
                ref_2d=ref_2d, ref_3d=ref_pnts,
                bev_h=h_bev, bev_w=w_bev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

        transformed_feature = query_embed.permute(0, 2, 1).view(
            bs, c, h_bev, w_bev).contiguous()

        # ── Return ──
        if self.is_train:
            return (self.lane_head(transformed_feature), heightmap,
                    self.lane_head_2d(sdf_feat), lane_heatmap, sdf_all, z_all_batch)
        else:
            return self.lane_head(transformed_feature), heightmap, None, lane_heatmap
