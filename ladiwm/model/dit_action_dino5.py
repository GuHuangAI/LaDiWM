import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from core.vivit import ViViT
# import clip
import math
from timm.models.vision_transformer import Attention, Mlp
import numpy as np
# from network.swin3d import SwinTransformer3D
from ladiwm.policy.vilt_modules.transformer_modules import TransformerDecoder
# add mae training
# 3D conv
# precond with ddm
# stack historical latent with input
# latent as condition rather than raw image
# simutaneous diffusion
# temporal + spatial transformer

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            img_size = img_size
        else:
            img_size = (img_size, img_size)
        if isinstance(patch_size, list) or isinstance(patch_size, tuple):
            patch_size = patch_size
        else:
            patch_size = (patch_size, patch_size)
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        T = C // 4
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = x.reshape(B, T, 4, H, W).transpose(1, 2)  # B, C, T, H, W
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            tube_size=2,
            hist_frame=4,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            img_size = img_size
        else:
            img_size = (img_size, img_size, img_size)
        if isinstance(patch_size, list) or isinstance(patch_size, tuple):
            patch_size = patch_size
        else:
            patch_size = (patch_size, patch_size, patch_size)
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tube_size = tube_size

        self.grid_size = (tube_size, img_size[0] // self.patch_size[1],
                          img_size[1] // self.patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        self.pos_embeder = SinePositionalEncoding3D()
        # self.proj = nn.Conv3d(in_chans, embed_dim,
        #                       kernel_size=(3, patch_size[1], patch_size[2]),
        #                       stride=(1, patch_size[1], patch_size[2]),
        #                       padding=(1, 0, 0),
        #                       bias=bias)
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(3, 3, 3),
                              stride=(1, patch_size[1], patch_size[2]),
                              padding=(1, 1, 1),
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        # T = C // 4
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        # x = x.reshape(B, T, 4, H, W).transpose(1, 2)  # B, C, T, H, W
        x = self.proj(x)
        pos = self.pos_embeder(x)
        # x = x + pos
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
            pos = pos.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x + pos
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate2(x, shift, scale):
    return x * (1 + scale) + shift

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x2, mask=None):
        B, N, C = x.shape
        _, N2, _ = x2.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, head, N, N2
        if mask is not None:
            # mask: B, N2
            mask = mask.unsqueeze(1).unsqueeze(1).expand(attn.shape)
            attn.masked_fill(mask < 0.1, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class ActionEmbedder(nn.Module):
#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super(ActionEmbedder, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size * 4, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True),
#         )
#         self.frequency_embedding_size = frequency_embedding_size
#
#     @staticmethod
#     def timestep_embedding(t, dim, max_period=10000):
#         """
#         Create sinusoidal timestep embeddings.
#         :param t: a 1-D Tensor of N indices, one per batch element.
#                           These may be fractional.
#         :param dim: the dimension of the output.
#         :param max_period: controls the minimum frequency of the embeddings.
#         :return: an (N, D) Tensor of positional embeddings.
#         """
#         # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
#         half = dim // 4
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=t.device)
#         args = (t.unsqueeze(-1).float() * freqs).flatten(-2)
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding
#
#     def forward(self, x):
#         # x: B, frame, 2
#         x = self.timestep_embedding(x, self.frequency_embedding_size)
#         x = self.mlp(x.flatten(1))
#         return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# class ActionEmbedder(nn.Module):
#     """
#     Embeds scalar timesteps into vector representations.
#     """
#     def __init__(self, hidden_size, input_size=8):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True),
#         )
#         # self.frequency_embedding_size = frequency_embedding_size
#
#     def forward(self, act):
#         # t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
#         a_emb = self.mlp(act)  # B, T, C
#         return a_emb

class ActionEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, input_size=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # self.frequency_embedding_size = frequency_embedding_size

    def forward(self, act):
        # t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # a_emb = self.mlp(act.flatten(1))
        a_emb = self.mlp(act)  # B, T, C
        return a_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x1, x2, t, c, mask=None, shape=None, type='spatial'):
        v_fea, v_fea2, cond_fea = c
        # x1 = rearrange(x1, '(b t) n c -> b (t n) c', b=shape[0])
        # x2 = rearrange(x2, '(b t) n c -> b (t n) c', b=shape[0])
        x1 = x1 + self.cross_attn(x1, v_fea, mask)
        x2 = x2 + self.cross_attn2(x2, v_fea2, mask)
        if type == 'spatial':
            x1 = rearrange(x1, 'b (t n) c -> (b t) n c', t=shape[1])
            x2 = rearrange(x2, 'b (t n) c -> (b t) n c', t=shape[1])
            t = repeat(t, 'b d -> (b t) d', t=shape[1])
            cond_fea = rearrange(cond_fea, 'b t d -> (b t) d')
            t = t + cond_fea
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
            shift_msa2, scale_msa2, gate_msa2, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(t).chunk(6, dim=-1)
            x1_tmp = modulate(self.norm1(x1), shift_msa, scale_msa)
            x2_tmp = modulate(self.norm2(x2), shift_msa2, scale_msa2)
            x = self.attn(torch.cat([x1_tmp, x2_tmp], dim=1))
            length = x.shape[1] // 2
            x1 = x1 + gate_msa.unsqueeze(1) * x[:, :length]
            x1 = x1 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x1), shift_mlp, scale_mlp))
            x2 = x2 + gate_msa2.unsqueeze(1) * x[:, length:]
            x2 = x2 + gate_mlp2.unsqueeze(1) * self.mlp2(modulate(self.norm4(x2), shift_mlp2, scale_mlp2))
        elif type=='temporal':
            x1 = rearrange(x1, 'b (t n) c -> (b n) t c', t=shape[1])
            x2 = rearrange(x2, 'b (t n) c -> (b n) t c', t=shape[1])
            t = repeat(t, 'b d -> (b n) d', n=shape[2])
            cond_fea = repeat(cond_fea, 'b t d -> (b n) t d', n=shape[2])
            t = t.unsqueeze(1) + cond_fea
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
            shift_msa2, scale_msa2, gate_msa2, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(t).chunk(6, dim=-1)
            x1_tmp = modulate2(self.norm1(x1), shift_msa, scale_msa)
            x2_tmp = modulate2(self.norm2(x2), shift_msa2, scale_msa2)
            x = self.attn(torch.cat([x1_tmp, x2_tmp], dim=1))
            length = x.shape[1] // 2
            x1 = x1 + gate_msa * x[:, :length]
            x1 = x1 + gate_mlp * self.mlp(modulate2(self.norm3(x1), shift_mlp, scale_mlp))
            x2 = x2 + gate_msa2 * x[:, length:]
            x2 = x2 + gate_mlp2 * self.mlp2(modulate2(self.norm4(x2), shift_mlp2, scale_mlp2))
        else:
            raise NotImplementedError
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        # shift_msa2, scale_msa2, gate_msa2, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(t).chunk(6, dim=-1)
        # x1_tmp = modulate(self.norm1(x1), shift_msa, scale_msa)
        # x2_tmp = modulate(self.norm2(x2), shift_msa2, scale_msa2)
        # x = self.attn(torch.cat([x1_tmp, x2_tmp], dim=1))
        # length = x.shape[1] // 2
        # x1 = x1 + gate_msa.unsqueeze(1) * x[:, :length]
        # x1 = x1 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x1), shift_mlp, scale_mlp))
        # x2 = x2 + gate_msa2.unsqueeze(1) * x[:, length:]
        # x2 = x2 + gate_mlp2.unsqueeze(1) * self.mlp2(modulate(self.norm4(x2), shift_mlp2, scale_mlp2))
        return x1, x2

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SinePositionalEncoding3D(nn.Module):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 # num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding3D, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        # self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, x):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].   # (B, N_view, H, W)
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].           # (B, N_view, num_feats*3, H, W)
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        # mask = mask.to(torch.int)
        # not_mask = 1 - mask  # logical_not
        B, C, N, H, W = x.size()
        num_feats = C // 3
        not_mask = torch.ones(B, N, H, W, device=x.device)
        n_embed = not_mask.cumsum(1, dtype=torch.float32)       # (B, N_view, H, W)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)       # (B, N_view, H, W)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)       # (B, N_view, H, W)

        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=x.device)    # (num_feats, )
        num_feats_final = C - num_feats * 2
        dim_t_final = torch.arange(
            num_feats_final, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / num_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]
        dim_t_final = self.temperature**(2 * (dim_t_final // 2) / num_feats_final)
        pos_n = n_embed[:, :, :, :, None] / dim_t       # (B, N_view, H, W, num_feats)      [pos_view/10000^(0/128), pos_view/10000^(0/128), pos_view/10000^(2/128), pos_view/10000^(2/128), ...]
        pos_x = x_embed[:, :, :, :, None] / dim_t       # (B, N_view, H, W, num_feats)      [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
        pos_y = y_embed[:, :, :, :, None] / dim_t_final       # (B, N_view, H, W, num_feats)      [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)     # (B, N_view, H, W, num_feats/2, 2) --> (B, N_view, H, W, num_feats)  num_feats: [sin(pos_view/10000^0/128), cos(pos_view/10000^0/128), sin(pos_view/10000^2/128), cos(pos_view/10000^2/128), ...]
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)     # (B, N_view, H, W, num_feats/2, 2) --> (B, N_view, H, W, num_feats)  num_feats: [sin(pos_x/10000^0/128), cos(pos_x/10000^0/128), sin(pos_x/10000^2/128), cos(pos_x/10000^2/128), ...]
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)     # (B, N_view, H, W, num_feats/2, 2) --> (B, N_view, H, W, num_feats)  num_feats: [sin(pos_y/10000^0/128), cos(pos_y/10000^0/128), sin(pos_y/10000^2/128), cos(pos_y/10000^2/128), ...]
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)    # (B, N_view, H, W, num_feats*3) --> (B, num_feats*3, N_view, H, W)

        return pos


class CompLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_heads, **block_kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        # self.linear = nn.Conv3d(hidden_size, 1 * patch_size * patch_size * out_channels,
        #                         kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.linear.bias, 0.)

    def forward(self, x, v, t, c, shape):
        B, T, N, _ = shape
        _, _, C = x.shape
        cond_fea = c
        x = x + self.cross_attn(x, v)
        x = rearrange(x, 'b (t n) c -> (b t) n c', t=shape[1])
        t = repeat(t, 'b d -> (b t) d', t=shape[1])
        cond_fea = rearrange(cond_fea, 'b t d -> (b t) d')
        t = t + cond_fea
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale) # B, T*N, C
        x = x.reshape(B, T, N, C)
        x = self.linear(x)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_heads, **block_kwargs):
        super().__init__()
        self.in_linear = nn.Linear(out_channels, hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        # self.linear = nn.Conv3d(hidden_size, 1 * patch_size * patch_size * out_channels,
        #                         kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y, t, c, shape):
        B, T, N, _ = shape
        _, _, C = x.shape
        cond_fea = c
        y = self.in_linear(y)
        x = x + self.cross_attn(x, y)
        x = rearrange(x, 'b (t n) c -> (b t) n c', t=shape[1])
        t = repeat(t, 'b d -> (b t) d', t=shape[1])
        cond_fea = rearrange(cond_fea, 'b t d -> (b t) d')
        t = t + cond_fea
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale) # B, T*N, C
        x = x.reshape(B, T, N, C)
        x = self.linear(x)
        return x

class Video3DConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, in_frames=4):
        super(Video3DConv, self).__init__()
        self.in_conv1 = nn.Sequential(nn.Conv3d(in_dim, hidden_dim//4,
                                 kernel_size=(3, 3, 3),
                                 stride=(2, 2, 2),
                                 padding=(1, 1, 1)
                                 ),
                                 nn.GroupNorm(8, hidden_dim//4)
                                )
        nn.init.xavier_uniform_(self.in_conv1[0].weight)
        nn.init.constant_(self.in_conv1[0].bias, 0)
        frame = in_frames // 2
        self.in_conv2 = nn.Sequential(
            nn.Conv3d(hidden_dim // 4, hidden_dim // 2,
                      kernel_size=(frame, 3, 3),
                      stride=(frame, 2, 2),
                      padding=(0, 1, 1)
                      ),
            nn.GroupNorm(8, hidden_dim//2),
        )
        nn.init.xavier_uniform_(self.in_conv2[0].weight)
        nn.init.constant_(self.in_conv2[0].bias, 0)
        self.in_conv3 = nn.Conv3d(hidden_dim // 2, hidden_dim,
                      kernel_size=(1, 3, 3),
                      stride=(1, 2, 2),
                      padding=(0, 1, 1))
        nn.init.xavier_uniform_(self.in_conv3.weight)
        nn.init.constant_(self.in_conv3.bias, 0)

    def forward(self, x):
        # x: b, c, t, h, w
        x = self.in_conv1(x)
        x = self.in_conv2(x)
        x = self.in_conv3(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1).contiguous()  # b, n, c
        return x

class VideoTransformer(nn.Module):
    def __init__(self, in_channel, dim, depth=8):
        super(VideoTransformer, self).__init__()
        self.global_token = nn.Parameter(torch.randn(1, 1, dim))
        self.in_layer = nn.Linear(in_channel, dim)
        self.transformer_layer = TransformerDecoder(
            input_size=dim,
            num_layers=depth,
            num_heads=8,
            head_output_size=dim//8,
            mlp_hidden_size=dim*2,
            dropout=0.1)
        # for _ in range(depth):
        #     layer =
        #     self.layers.append(layer)

    def forward(self, x):
        # x: b t n c
        x = self.in_layer(x)
        B, T, N, C = x.shape
        x = rearrange(x, 'b t n c -> (b n) t c', b=B)
        x = torch.cat((self.global_token.repeat(x.shape[0], 1, 1), x), dim=1)
        x = self.transformer_layer(x)[:, 0]
        x = rearrange(x, '(b n) c -> b n c', b=B)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, img_size=480, down_ratio=8, patch_size=16, mlp_ratio=4.0, n_layers=12,
                 dim=768, num_heads=8, vivit_load_from=None, load_from=None, load_from_ema=False, proj_drop=0.1,
                 attn_drop=0.1, tube_size=4, hist_frame=4, **kwargs):
        super(DiffusionTransformer, self).__init__()
        # self.video_encoder = ViViT(num_frames=4,
        #       img_size=img_size,
        #       patch_size=16,
        #       embed_dims=768,
        #       tube_size=tube_size,
        #       load_from=vivit_load_from,
        #       attention_type='fact_encoder',
        #       weights_from='kinetics',
        #       use_learnable_pos_emb=False,
        #       return_cls_token=False)
        # self.video_encoder = Video3DConv(in_dim=3, hidden_dim=dim)
        # self.video_encoder = SwinTransformer3D2(
        #         in_chans=3,
        #         embed_dim=128,
        #         num_heads=[4, 8, 16, 32],
        #         # num_heads=[4, 8, 16],
        #         patch_size=(2, 4, 4),
        #         window_size=(2, 7, 7),
        #         depths=[2, 2, 18, 2],
        #         # depths=[2, 2, 18],
        #         drop_path_rate=0.1,
        #         out_channel=dim,
        #         pretrained=kwargs.get("pretrained_swin", None),
        #     )
        self.video_encoder = VideoTransformer(in_channel=in_channel, dim=dim)
        self.video_encoder2 = VideoTransformer(in_channel=in_channel, dim=dim)
        # self.proj_v = nn.Linear(768, dim, bias=False)
        # self.text_encoder, preprocess = clip.load('ViT-B/16', device='cpu')
        # self.action_encoder = ActionEmbedder(dim)
        self.time_encoder = TimestepEmbedder(dim)
        # self.text_proj = nn.Linear(768, dim)

        self.action_encoder = ActionEmbedder(dim, input_size=7)

        self.num_heads = num_heads
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.patch_size = patch_size
        self.tube_size = tube_size
        self.img_size = img_size
        # if isinstance(img_size, list) or isinstance(img_size, tuple):
        # self.x_embedder = PatchEmbed3D([img_size[0]//down_ratio, img_size[1]//down_ratio],
        #                              patch_size, in_channel, dim, tube_size, hist_frame, bias=True)
        self.x_embedder = nn.Linear(in_channel, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size * tube_size, dim), requires_grad=False)
        self.x_embedder2 = nn.Linear(in_channel, dim)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, img_size * tube_size, dim), requires_grad=False)
        # self.x_embedder2 = PatchEmbed3D([img_size[0], img_size[1]],
        #                                patch_size * down_ratio, 1, dim, tube_size, hist_frame, bias=True)
        # else:
        #     self.x_embedder = PatchEmbed3D(img_size // down_ratio, patch_size, in_channel,
        #                                  dim, bias=True)
        # num_patches = self.x_embedder.num_patches
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop) for _ in range(n_layers)
        ])
        self.component1 = CompLayer(dim, patch_size, self.in_channel, num_heads)
        self.component2 = CompLayer(dim, patch_size, self.in_channel, num_heads)
        self.final_layer = FinalLayer(dim, patch_size, self.in_channel, num_heads)
        self.final_layer2 = FinalLayer(dim, patch_size, self.in_channel, num_heads)
        self.final_layer3 = FinalLayer(dim, patch_size, self.in_channel, num_heads)
        self.final_layer4 = FinalLayer(dim, patch_size, self.in_channel, num_heads)
        self.initialize_weights(load_from, load_from_ema=load_from_ema)

    def initialize_weights(self, load_from=None, load_from_ema=False):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.tube_size, self.img_size))
        # pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.pos_embed.shape[1]))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed2.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        w = self.x_embedder2.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder2.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize cond_proj layer:
        # nn.init.normal_(self.text_proj.weight, std=0.02)
        # nn.init.normal_(self.proj_v.weight, std=0.02)

        # Initialize action embedding MLP:
        # nn.init.normal_(self.action_encoder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.action_encoder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_encoder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_encoder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.action_encoder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.action_encoder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation2[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation2[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer2.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer2.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer2.linear.weight, 0)
        nn.init.constant_(self.final_layer2.linear.bias, 0)
        nn.init.constant_(self.final_layer3.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer3.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer3.linear.weight, 0)
        nn.init.constant_(self.final_layer3.linear.bias, 0)
        nn.init.constant_(self.final_layer4.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer4.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer4.linear.weight, 0)
        nn.init.constant_(self.final_layer4.linear.bias, 0)

        if load_from is not None:
            sd = torch.load(load_from, map_location="cpu")
            if 'ema' in list(sd.keys()) and load_from_ema:
                sd = sd['ema']
                new_sd = {}
                for k in sd.keys():
                    if k.startswith("ema_model."):
                        new_k = k[10:]  # remove ema_model.
                        new_sd[new_k] = sd[k]
                    else:
                        new_sd[k] = sd[k]
                sd = new_sd
            else:
                if "model" in list(sd.keys()):
                    sd = sd["model"]
            # keys = list(sd.keys())
            # for k in keys:
            #     for ik in ignore_keys:
            #         if k.startswith(ik):
            #             print("Deleting key {} from state_dict.".format(k))
            #             del sd[k]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"Restored from {load_from} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**3 * C)
        imgs: (N, H, W, C)
        """
        c = self.in_channel
        p = self.x_embedder.patch_size[0]
        img_size = self.x_embedder.img_size
        # t, h, w = 5, img_size[0] // p, img_size[1] // p
        t, h, w = self.x_embedder.grid_size
        # t = 1
        # h = w = int(x.shape[1] ** 0.5)
        # assert t * h * w == x.shape[1]

        # x = x.reshape(shape=(x.shape[0], t, h, w, p, p, p, c))
        # x = torch.einsum('nthwopqc->nctohpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], c, t * p, h * p, w * p))
        # imgs = imgs.transpose(1, 2)
        x = x.reshape(shape=(x.shape[0], 1, p, p, c, t, h, w))
        x = torch.einsum('nopqcthw->nctohpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, t, h * p, w * p))
        return imgs  # B, C, T, H, W

    def unpatchify2(self, x):
        """
        x: (N, T, patch_size**3 * C)
        imgs: (N, H, W, C)
        """
        c = 1
        p = self.x_embedder2.patch_size[0]
        img_size = self.x_embedder2.img_size
        # t, h, w = 5, img_size[0] // p, img_size[1] // p
        t, h, w = self.x_embedder2.grid_size
        # t = 1
        # h = w = int(x.shape[1] ** 0.5)
        # assert t * h * w == x.shape[1]

        # x = x.reshape(shape=(x.shape[0], t, h, w, p, p, p, c))
        # x = torch.einsum('nthwopqc->nctohpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], c, t * p, h * p, w * p))
        # imgs = imgs.transpose(1, 2)
        x = x.reshape(shape=(x.shape[0], 1, p, p, c, t, h, w))
        x = torch.einsum('nopqcthw->nctohpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, t, h * p, w * p))
        return imgs  # B, C, T, H, W

    def forward(self, x_dino, x_siglip, context, time, mask_his=None, *args, **kwargs):
        x_in1 = x_dino.to(torch.float32)  # B, T, N, C
        x_in2 = x_siglip.to(torch.float32)  # B, T, N, C
        shape = x_in1.shape
        # print(shape)
        time = time.to(torch.float32)
        x_clone1 = x_in1.clone()
        x_clone2 = x_in2.clone()
        sigma = time.reshape(x_in1.shape[0], *((1,) * (len(x_in1.shape) - 1)))
        precond = kwargs.get("precond", False)
        if precond:
            c_skip1 = (sigma - 1) / (sigma ** 2 + (sigma - 1) ** 2)
            c_out1 = sigma / (sigma ** 2 + (sigma - 1) ** 2).sqrt()
            c_skip2 = sigma / (sigma ** 2 + (sigma - 1) ** 2)
            c_out2 = (1 - sigma) / (sigma ** 2 + (sigma - 1) ** 2).sqrt()
        # x = self.x_embedder(x_in).reshape(shape[0]*shape[1], x_in.shape[2], -1) + self.pos_embed
        x1 = self.x_embedder(x_in1).reshape(shape[0], shape[1] * shape[2], -1) + self.pos_embed  # B, T*N, C
        x2 = self.x_embedder2(x_in2).reshape(shape[0], shape[1] * shape[2], -1) + self.pos_embed2  # B, T*N, C
        # x = self.x_embedder(x_in)
        # shape = self.x_embedder.grid_size
        his_latent1, his_latent2, text, action = context
        # video = video.transpose(1, 2)  # b, c, t, h, w

        # mask_his = mask_his.transpose(1, 2)  # b, 1, t, h, w
        v_fea = self.video_encoder(his_latent1)  # b n c
        v_fea2 = self.video_encoder2(his_latent2)  # b n c

        # v_fea = self.proj_v(v_fea)
        # a_fea = self.action_encoder(action * 100)
        # act_mask = kwargs.get('act_mask', None)
        # if act_mask is None:
        #     act_mask = torch.bernoulli(torch.zeros(a_fea.shape[0]) + 0.1).to(a_fea.device)
        #     act_mask = act_mask[:, None]
        #     act_mask = act_mask.repeat(1, a_fea.shape[1])
        #     act_mask = 1 - act_mask
        # else:
        #     act_mask = act_mask[:, None]
        #     act_mask = act_mask.repeat(1, a_fea.shape[1])
        #     act_mask = 1 - act_mask
        # a_fea = a_fea * act_mask
        # text_fea = self.text_encoder.encode_text(text)
        time_fea = self.time_encoder(time.log()) # b, d
        act_fea = self.action_encoder(action)  # b, t, d
        # text_fea = self.text_proj(text)
        is_training = kwargs.get("is_training", True)
        cond_fea = act_fea
        # if is_training:
        #     rands = torch.rand(x.shape[0], 1, device=x.device)
        #     rands = (rands >= 0.2).to(torch.float32)
        #     cond_fea = rands * act_fea + (1 - rands) * text_fea
        # else:
        #     act_mask = kwargs.get("use_action", torch.ones(x_in.shape[0], device=x_in.device)).unsqueeze(1)
        #     cond_fea = act_mask * act_fea + (1 - act_mask) * text_fea
            # if kwargs.get("use_action", True):
            #     cond_fea = act_fea
            # else:
            #     cond_fea = text_fea
        # x1 = rearrange(x1, 'b (t n) c -> (b t) n c', t=shape[1])
        # x2 = rearrange(x2, 'b (t n) c -> (b t) n c', t=shape[1])
        cond = [v_fea, v_fea2, cond_fea]
        for i in range(0, len(self.blocks), 2):
        # for block in self.blocks:
            block1, block2 = self.blocks[i], self.blocks[i+1]
            x1, x2 = block1(x1, x2, time_fea, cond, shape=shape, type='spatial')
            x1 = rearrange(x1, '(b t) n c -> b (t n) c', b=shape[0])
            x2 = rearrange(x2, '(b t) n c -> b (t n) c', b=shape[0])
            x1, x2 = block2(x1, x2, time_fea, cond, shape=shape, type='temporal')
            x1 = rearrange(x1, '(b n) t c -> b (t n) c', b=shape[0])
            x2 = rearrange(x2, '(b n) t c -> b (t n) c', b=shape[0])

        y_tmp1 = self.component1(x1, v_fea, time_fea, cond_fea, shape=shape)
        y_tmp2 = self.component2(x2, v_fea2, time_fea, cond_fea, shape=shape)
        y = self.final_layer(x1, y_tmp2.flatten(1, 2), time_fea, cond_fea,
                             shape=shape)  # (N, T, patch_size ** 2 * out_channels)
        # y = self.unpatchify(y)  # (N, 4, T, H, W)
        y2 = self.final_layer2(x1, y_tmp2.flatten(1, 2), time_fea, cond_fea,
                               shape=shape)  # (N, T, patch_size ** 2 * out_channels)
        # y2 = self.unpatchify(y2)  # (N, T*4, H, W)
        y3 = self.final_layer3(x2, y_tmp1.flatten(1, 2), time_fea, cond_fea,
                             shape=shape)  # (N, T, patch_size ** 2 * out_channels)
        y4 = self.final_layer4(x2, y_tmp1.flatten(1, 2), time_fea, cond_fea,
                             shape=shape)  # (N, T, patch_size ** 2 * out_channels)

        if precond:
            y = c_skip1 * x_clone1 + c_out1 * y
            y2 = c_skip2 * x_clone1 + c_out2 * y2
            y3 = c_skip1 * x_clone2 + c_out1 * y3
            y4 = c_skip2 * x_clone2 + c_out2 * y4
        return y, y2, y3, y4, y_tmp1, y_tmp2

if __name__ == '__main__':
    x = torch.rand(1, 4, 256, 4)
    text = torch.randint(4000, (1, 77))
    act = torch.rand(1, 4, 7)
    his = torch.rand(1, 4, 3, 144, 256)
    time = torch.rand(1,)

    model = DiffusionTransformer(
        in_channel=4, img_size=256, patch_size=2, vivit_load_from='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/pre_weight/vivit_model.pth',
    )
    with torch.no_grad():
        y = model(x, [his, text, act], time)
    pause = 0