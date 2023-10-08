import re
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = False,
) -> torch.Tensor:
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is `[x, y]`, the results will  # noqa 
            be `[pos(y), pos(x)]`. Defaults: True.
    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res


@TRANSFORMER.register_module()
class Uni3DETRTransformer(BaseModule):
    def __init__(self,
                 decoder=None,
                 fp16_enabled=False,
                 **kwargs):
        super(Uni3DETRTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.d_model = 256
        
        self.init_layers()
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

    def init_layers(self):
        return 

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, UniCrossAtten):
                m.init_weight()

    @auto_fp16(apply_to=('pts_value', 'query_embed'))
    def forward(self,
                pts_value,
                query_embed,
                num_query,
                reg_branches=None,
                **kwargs):
        
        assert query_embed is not None
        bs = pts_value.shape[0]
        reference_points = query_embed[..., self.d_model:]
        query = query_embed[..., :self.d_model]
        query_pos = query_embed
        init_reference_out = reference_points.sigmoid() 
        query = query.permute(1, 0, 2)
        
        nq = num_query
        ng = int(len(query) // nq)
        inter_states = []
        inter_references = []
        for g in range(ng):
            inter_states1, inter_references1 = self.decoder(
                query=query[g*nq:(g+1)*nq],
                key=None,
                value=pts_value,
                query_pos=None,
                reference_points=reference_points[:, g*nq:(g+1)*nq, :],
                reg_branches=reg_branches,
                **kwargs)
            inter_states.append(inter_states1)
            inter_references.append(inter_references1)
        inter_states = torch.cat(inter_states, 1)
        inter_references = torch.cat(inter_references, 2)

        inter_references_out = inter_references.sigmoid()
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Uni3DETRTransformerDecoder(TransformerLayerSequence):
    """
    Implements the decoder in UVTR transformer.
    """
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Uni3DETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.d_model = d_model = 256
        self.query_scale = MLP(d_model, d_model, d_model, 3)
        self.ref_point_head = MLP(384, d_model, d_model, 3)

    def forward(self,
                query,
                key, value, query_pos,
                reference_points=None,
                reg_branches=None,
                attn_masks=None,
                **kwargs):
        """
        Forward function for `UniTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_points_nodetach = []

        assert query_pos is None
        bs = query.shape[1]
        #reference_points = reference_points[None].repeat(bs, 1, 1)

        for lid, layer in enumerate(self.layers):
            query_sine_embed = get_sine_pos_embed(reference_points.sigmoid()) # bs, nq, 256*2 
            raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
            pos_scale = self.query_scale(output) if lid != 0 else 1
            if lid != 0:
                pos_scale = pos_scale.permute(1, 0, 2)
            query_pos = pos_scale * raw_query_pos
            query_pos = query_pos.permute(1, 0, 2)

            output = layer(
                output,
                key, value, query_pos=query_pos,
                reference_points=reference_points,
                **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                
                assert reference_points.shape[-1] == 3
                # tmp: (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + reference_points[..., :2]
                new_reference_points[..., 2:3] = tmp[..., 4:5] + reference_points[..., 2:3]                    
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class UniCrossAtten(BaseModule):
    """
    Cross attention module in UVTR. 
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_points=1,
                 num_sweeps=1,
                 cam_sweep_feq=12,
                 voxel_range=(0,0,0),
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 fp16_enabled=False):
        super(UniCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_sweeps = num_sweeps
        self.cam_sweep_time = 1.0 / cam_sweep_feq
        self.voxel_range = torch.Tensor(voxel_range)
        self.attention_weights = nn.Linear(embed_dims, num_points)

        #self.sampling_offsets = nn.Linear(embed_dims, num_points * 3)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.init_weight()
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('query', 'key'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of UniCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # ATTENTION: reference_points is decoupled from sigmoid function!
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos
        
        pts_value = value
        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        # change to (bs, num_query, num_points)
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.sigmoid()

        # normalize X, Y, Z to [-1,1]
        reference_points_voxel = (reference_points.sigmoid() - 0.5) * 2

        embed_uni = []
        
        # shape: (N, C, D, H, W)
        # without height
        if len(pts_value.shape) == 4:
            reference_points_voxel = reference_points_voxel.view(-1,1,*reference_points_voxel.shape[-2:])[...,:2]
        else:
            pts_value = pts_value.view(-1, *pts_value.shape[2:])
            reference_points_voxel = reference_points_voxel.view(-1, 1, 1, *reference_points_voxel.shape[-2:])
        # sample image feature in voxel space
        embed_pts = F.grid_sample(pts_value, reference_points_voxel)
        embed_pts = embed_pts.reshape(len(query), -1, embed_pts.shape[1], embed_pts.shape[-1])
        embed_pts = embed_pts.permute(0, 3, 2, 1)
        embed_uni.append(embed_pts)
        # concat embeddings different modalities
        embed_uni = torch.cat(embed_uni, dim=-1)
        output = embed_uni * attention_weights.unsqueeze(-2)

        output = output.sum(-1)

        output = output.permute(1, 0, 2)
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(reference_points).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat
