import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange, repeat

from ladiwm.model import *
from ladiwm.model.track_patch_embed import TrackPatchEmbed
from ladiwm.policy.vilt_modules.transformer_modules import *
from ladiwm.policy.vilt_modules.rgb_modules import *
from ladiwm.policy.vilt_modules.language_modules import *
from ladiwm.policy.vilt_modules.extra_state_modules import ExtraModalityTokens
from ladiwm.policy.vilt_modules.policy_head import *
from ladiwm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video
from ladiwm.utils.build_utils import construct_class_by_name
from ladiwm.model.dino_vit import vit_base
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
import random
from ladiwm.utils.transform_utils import axisangle2quat_torch, quat2axisangle_torch, quat2mat_torch, mat2quat_torch


###############################################################################
#
# A ViLT Policy
#
###############################################################################

class DINO_Processor(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.trans = transforms.Compose([
            lambda x: 255.0 * x, # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear')
        x = self.trans(x)
        return x

class Siglip_Processor(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.trans = transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear')
        x = self.trans(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.mlp(emb)
        return emb

# def axis_angle_to_rotation_matrix(axis_angle):
#     theta = torch.norm(axis_angle, dim=1, keepdim=True)  # 旋转角度
#     k = axis_angle / (theta + 1e-8)  # 单位化旋转轴
#
#     # 构造反对称矩阵 K
#     K = torch.zeros(axis_angle.size(0), 3, 3, device=axis_angle.device)
#     K[:, 0, 1] = -k[:, 2]
#     K[:, 0, 2] = k[:, 1]
#     K[:, 1, 0] = k[:, 2]
#     K[:, 1, 2] = -k[:, 0]
#     K[:, 2, 0] = -k[:, 1]
#     K[:, 2, 1] = k[:, 0]
#
#     I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
#     R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * torch.bmm(K, K)
#     return R


# def quaternion_to_matrix(quaternions):
#     """
#     Convert rotations given as quaternions to rotation matrices.
#
#     Args:
#         quaternions: quaternions with real part first,
#             as tensor of shape (..., 4).
#
#     Returns:
#         Rotation matrices as tensor of shape (..., 3, 3).
#     """
#     r, i, j, k = torch.unbind(quaternions, -1)
#     two_s = 2.0 / (quaternions * quaternions).sum(-1)
#
#     o = torch.stack(
#         (
#             1 - two_s * (j * j + k * k),
#             two_s * (i * j - k * r),
#             two_s * (i * k + j * r),
#             two_s * (i * j + k * r),
#             1 - two_s * (i * i + k * k),
#             two_s * (j * k - i * r),
#             two_s * (i * k - j * r),
#             two_s * (j * k + i * r),
#             1 - two_s * (i * i + j * j),
#         ),
#         -1,
#     )
#     return o.reshape(quaternions.shape[:-1] + (3, 3))


# def axis_angle_to_quaternion(axis_angle):
#     """
#     Convert rotations given as axis/angle to quaternions.
#
#     Args:
#         axis_angle: Rotations given as a vector in axis angle form,
#             as a tensor of shape (..., 3), where the magnitude is
#             the angle turned anticlockwise in radians around the
#             vector's direction.
#
#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     """
#     angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
#     half_angles = 0.5 * angles
#     eps = 1e-6
#     small_angles = angles.abs() < eps
#     sin_half_angles_over_angles = torch.empty_like(angles)
#     sin_half_angles_over_angles[~small_angles] = (
#             torch.sin(half_angles[~small_angles]) / angles[~small_angles]
#     )
#     # for x small, sin(x/2) is about x/2 - (x/2)^3/6
#     # so sin(x/2)/x is about 1/2 - (x*x)/48
#     sin_half_angles_over_angles[small_angles] = (
#             0.5 - (angles[small_angles] * angles[small_angles]) / 48
#     )
#     quaternions = torch.cat(
#         [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
#     )
#     return quaternions
#
#
# def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
#     """
#     Returns torch.sqrt(torch.max(0, x))
#     but with a zero subgradient where x is 0.
#     """
#     ret = torch.zeros_like(x)
#     positive_mask = x > 0
#     ret[positive_mask] = torch.sqrt(x[positive_mask])
#     return ret
#
#
# def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to quaternions.
#
#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).
#
#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     """
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
#
#     batch_dim = matrix.shape[:-2]
#     m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
#         matrix.reshape(*batch_dim, 9), dim=-1
#     )
#
#     q_abs = _sqrt_positive_part(
#         torch.stack(
#             [
#                 1.0 + m00 + m11 + m22,
#                 1.0 + m00 - m11 - m22,
#                 1.0 - m00 + m11 - m22,
#                 1.0 - m00 - m11 + m22,
#             ],
#             dim=-1,
#         )
#     )
#
#     # we produce the desired quaternion multiplied by each of r, i, j, k
#     quat_by_rijk = torch.stack(
#         [
#             torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
#             torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
#             torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
#             torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
#         ],
#         dim=-2,
#     )
#
#     # We floor here at 0.1 but the exact level is not important; if q_abs is small,
#     # the candidate won't be picked.
#     # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
#     quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))
#
#     # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
#     # forall i; we pick the best-conditioned one (with the largest denominator)
#
#     return quat_candidates[
#            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
#            ].reshape(*batch_dim, 4)
#
#
# def quaternion_to_axis_angle(quaternions):
#     """
#     Convert rotations given as quaternions to axis/angle.
#
#     Args:
#         quaternions: quaternions with real part first,
#             as tensor of shape (..., 4).
#
#     Returns:
#         Rotations given as a vector in axis angle form, as a tensor
#             of shape (..., 3), where the magnitude is the angle
#             turned anticlockwise in radians around the vector's
#             direction.
#     """
#     norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
#     half_angles = torch.atan2(norms, quaternions[..., :1])
#     angles = 2 * half_angles
#     eps = 1e-6
#     small_angles = angles.abs() < eps
#     sin_half_angles_over_angles = torch.empty_like(angles)
#     sin_half_angles_over_angles[~small_angles] = (
#             torch.sin(half_angles[~small_angles]) / angles[~small_angles]
#     )
#     # for x small, sin(x/2) is about x/2 - (x/2)^3/6
#     # so sin(x/2)/x is about 1/2 - (x*x)/48
#     sin_half_angles_over_angles[small_angles] = (
#             0.5 - (angles[small_angles] * angles[small_angles]) / 48
#     )
#     return quaternions[..., 1:] / sin_half_angles_over_angles
#
#
# def rotation_matrix_to_axis_angle(rot_matrices):
#     """
#     将旋转矩阵转换为轴角表示。
#
#     参数:
#     rot_matrices: 形状为 (N, 3, 3) 的张量，表示 N 个旋转矩阵
#
#     返回:
#     axis_angles: 形状为 (N, 3) 的张量，表示 N 个轴角对应的 (轴, 角度)
#     """
#     return quaternion_to_axis_angle(matrix_to_quaternion(rot_matrices))

class BCViLTPolicyDiff_DINO_SIGLIP_WM5(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, obs_cfg, img_encoder_cfg, language_encoder_cfg, extra_state_encoder_cfg, track_cfg,
                 spatial_transformer_cfg, temporal_transformer_cfg,
                 policy_head_cfg, load_path=None, sampling_step=1, dino_preweight=None):
        super().__init__()

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. encode language (spatial)
        self.language_encoder_spatial = self._setup_language_encoder(output_size=self.spatial_embed_size, **language_encoder_cfg)

        # 3. Track Transformer module
        self._setup_track(**track_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg)

        # 6. encode language (temporal), this will also act as the TEMPORAL_TOKEN, i.e., CLS token for action prediction
        self.language_encoder_temporal = self._setup_language_encoder(output_size=self.temporal_embed_size, **language_encoder_cfg)

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
        self._setup_policy_head(**policy_head_cfg)

        # 9. define decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.temporal_embed_size,
            nhead=8,
            dim_feedforward=4 * self.temporal_embed_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=6
        )
        self.input_proj_act = nn.Linear(policy_head_cfg['output_size'][-1], self.temporal_embed_size)
        nn.init.normal_(self.input_proj_act.weight, mean=0.0, std=0.02)
        self.pe_query = nn.Parameter(torch.zeros(1, policy_head_cfg['output_size'][0], self.temporal_embed_size))
        nn.init.normal_(self.pe_query, mean=0.0, std=0.02)
        self.sampling_step = sampling_step
        # time embedding
        self.time_embed = SinusoidalPosEmb(self.temporal_embed_size)
        # self.dino = vit_base(patch_size=14, num_register_tokens=0,
        #                      img_size=526,
        #                      init_values=1.0,
        #                      block_chunks=0,
        #                      pre_weight='/data1/huang/pre_weight/dinov2_vitb14_pretrain.pth')

        self.dino = vit_base(patch_size=14, num_register_tokens=0,
                             img_size=526,
                             init_values=1.0,
                             block_chunks=0,
                             pre_weight=dino_preweight)
        self.dino_processor = DINO_Processor((126, 126))
        self.siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        # self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n")
        self.siglip_processor = Siglip_Processor((144, 144))

        for p in self.dino.parameters():
            p.requires_grad = False

        for p in self.siglip.parameters():
            p.requires_grad = False
        if load_path is not None:
            self.load(load_path)
            # self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")

    def _process_obs_shapes(self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len
        # define buffer queue for encoded latent features
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)
        self.joint_state_queue = deque(maxlen=max_seq_len)
        self.gripper_state_queue = deque(maxlen=max_seq_len)
        self.ee_pos_queue = deque(maxlen=max_seq_len)
        self.ee_state_queue = deque(maxlen=max_seq_len)

    def _setup_image_encoder(self, network_name, patch_size, embed_size, no_patch_embed_bias):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            input_shape = [input_shape[0], 4, input_shape[2] // 8, input_shape[3] // 8]
            # self.image_encoders.append(eval(network_name)(input_shape=input_shape, patch_size=patch_size,
            #                                               embed_size=self.spatial_embed_size,
            #                                               no_patch_embed_bias=no_patch_embed_bias))
            proj_layer = nn.Linear(768, self.spatial_embed_size)
            nn.init.normal_(proj_layer.weight, 0., 0.02)
            nn.init.constant_(proj_layer.bias, 0.)
            self.image_encoders.append(proj_layer)
        self.image_encoders = nn.ModuleList(self.image_encoders)
        self.image_frame = input_shape[0]

        # self.img_num_patches = sum([x.num_patches for x in self.image_encoders])
        self.img_num_patches = 81*2*2


    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_track(self, track_fn, policy_track_patch_size=None, use_zero_track=False):
        """
        track_fn: path to the track model
        policy_track_patch_size: The patch size of TrackPatchEmbedding in the policy, if None, it will be assigned the same patch size as TrackTransformer by default
        use_zero_track: whether to zero out the tracks (ie use only the image)
        """
        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        self.use_zero_track = use_zero_track
        #
        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        # # track_cls = eval(track_cfg.model_name)
        # # self.track = track_cls(**track_cfg.model_cfg)
        vae = construct_class_by_name(**track_cfg.vae_cfg)
        transformer = construct_class_by_name(**track_cfg.trans_cfg)
        model_kwargs = OmegaConf.to_container(track_cfg.model_cfg, resolve=True)
        model_kwargs.update({'vae': vae})
        model_kwargs.update({'transformer': transformer})
        self.track = construct_class_by_name(**model_kwargs)
        del model_kwargs
        # # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = 32
        self.num_track_ts = 16
        self.policy_track_patch_size = 4 if policy_track_patch_size is None else policy_track_patch_size


        # self.track_proj_encoder = TrackPatchEmbed(
        #     num_track_ts=self.policy_num_track_ts,
        #     num_track_ids=self.num_track_ids,
        #     patch_size=self.policy_track_patch_size,
        #     in_dim=2 + self.num_views,  # X, Y, one-hot view embedding
        #     embed_dim=self.spatial_embed_size)

        self.track_id_embed_dim = 16
        # self.num_track_patches_per_view = self.track_proj_encoder.num_patches_per_track
        # self.num_track_patches = self.num_track_patches_per_view * self.num_views

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(torch.randn(1, 1, self.spatial_embed_size))  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(torch.randn(1, self.img_num_patches, self.spatial_embed_size))
        # self.spatial_cls_token = nn.Parameter(torch.randn(1, self.image_frame, self.spatial_embed_size))
        # track_patch_pos_embed = nn.Parameter(torch.randn(1, self.num_track_patches, self.spatial_embed_size-self.track_id_embed_dim))
        # modality_embed = nn.Parameter(
        #     torch.randn(1, len(self.image_encoders) + self.num_views + 1, self.spatial_embed_size)
        # )  # IMG_PATCH_TOKENS + TRACK_PATCH_TOKENS + SENTENCE_TOKEN
        modality_embed = nn.Parameter(
            torch.randn(1, len(self.image_encoders) + 1, self.spatial_embed_size)
        )  # IMG_PATCH_TOKENS + SENTENCE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        # self.register_parameter("track_patch_pos_embed", track_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        # modality_idx = []
        # for i, encoder in enumerate(self.image_encoders):
        #     modality_idx += [i] * encoder.num_patches
        # # for i in range(self.num_views):
        # #     modality_idx += [modality_idx[-1] + 1] * self.num_track_ids * self.num_track_patches_per_view  # for track embedding
        # modality_idx += [modality_idx[-1] + 1]  # for sentence embedding
        # self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_pos" in self.extra_state_keys),
                use_ee2=("ee_states" in self.extra_state_keys),
                **extra_state_encoder_cfg
            )

    def _setup_spatial_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout,
                                   spatial_downsample, spatial_downsample_embed_size, use_language_token=True):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )
        # self.time_img_transformer = TransformerDecoder(
        #     input_size=self.spatial_embed_size,
        #     num_layers=num_layers,
        #     num_heads=num_heads,
        #     head_output_size=head_output_size,
        #     mlp_hidden_size=mlp_hidden_size,
        #     dropout=dropout, )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(self.spatial_embed_size, self.temporal_embed_size)
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout, use_language_token=True):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(input_size=self.temporal_embed_size)

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,)
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        policy_head_kwargs["input_size"] \
            = self.temporal_embed_size #* policy_head_kwargs['input_frame'] # + self.num_views * self.policy_num_track_ts * self.policy_num_track_ids * 2

        action_shape = policy_head_kwargs["output_size"]
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        # self.pred_frame = policy_head_kwargs['pred_frame']
        # self.policy_head = eval(network_name)(**policy_head_kwargs)
        self.policy_head1 = nn.Sequential(
            nn.Linear(policy_head_kwargs["input_size"], 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_shape[-1])
        )
        for m in self.policy_head1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)
        self.policy_head2 = nn.Sequential(
            nn.Linear(policy_head_kwargs["input_size"], 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_shape[-1])
        )
        for m in self.policy_head2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def preprocess(self, obs, track, action):
        """
        Preprocess observations, according to an observation dictionary.
        Return the feature and state.
        """
        b, v, t, c, h, w = obs.shape

        action = action.reshape(b, *self.act_shape)

        # obs = self._preprocess_rgb(obs)
        obs = obs / 255.
        # obs = 2 * obs - 1.

        return obs, track, action

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        # rgb = self.img_normalizer(rgb / 255.)
        rgb = rgb / 255.
        # rgb = 2 * rgb - 1.
        return rgb

    def _get_view_one_hot(self, tr):
        """ tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
        b, v, t, tl, n, d = tr.shape
        tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
        one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(tr.shape[0], 1, 1)
        tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
        tr_view = rearrange(tr_view, "(b t tl n) v c -> b v t tl n c", b=b, v=v, t=t, tl=tl, n=n, c=d + v)
        return tr_view

    def track_encode(self, track_obs, task_emb):
        """
        Args:
            track_obs: b v t tt_fs c h w
            task_emb: b e
        Returns: b v t track_len n 2
        """
        assert self.num_track_ids == 32
        b, v, t, *_ = track_obs.shape

        if self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

            grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
            grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts)
            grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
        tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)  # (b t n) (v patch_num) d

        return tr, _recon_tr

    def pca(self, x, dim):
        u, s, v = torch.pca_lowrank(x, q=dim, niter=3)
        out = torch.matmul(x, v[:, :, :dim])
        return out

    @torch.no_grad()
    def extract_latent(self, x):
        # x: b t c h w
        b, t = x.shape[:2]
        x = x.flatten(0, 1)
        ret_dict = self.dino.forward_features(self.dino_processor(x))
        h1 = ret_dict["x_norm_patchtokens"]  # b*t, 256, 768
        # h1 = self.pca(h1, dim=32)
        h2 = self.siglip.get_image_features(self.siglip_processor(x), interpolate_pos_encoding=True)
        h2 = h2[0]  # b*t, 1, 768
        # h2 = self.pca(h2, dim=32)
        latent = torch.cat([h1, h2], dim=1)  # b*t, 257, 768
        latent = latent.reshape(b, t, *latent.shape[1:])
        return latent.detach()

    def image_encode(self, obs, wm_act, use_action):
        img_encoded = []
        for view_idx in range(self.num_views):
            obs_tmp = obs[:, view_idx, ...]  # b t c h w
            b, t = obs_tmp.shape[:2]
            with torch.no_grad():
                obs_lat = self.extract_latent(obs_tmp)  # b t n c
            cond = [obs_lat[:, -4:].clone(), None, wm_act]
            imagined_lat = self.track.sample_from_latent(cond=cond)
            tmp = obs_lat[:, -1:].clone().repeat(1, self.track.transformer.tube_size, 1, 1)
            imagined_lat = imagined_lat * use_action.reshape(use_action.shape[0], 1, 1, 1) + \
                           tmp * (1 - use_action).reshape(use_action.shape[0], 1, 1, 1)
            obs_lat = torch.cat([obs_lat[:, -self.max_seq_len:], imagined_lat], dim=1)  # b t n c
            # obs_lat = obs_lat.reshape(b, t, obs_lat.shape[-2], obs_lat.shape[-1])
            # obs_lat = rearrange(obs_lat, '(b t) n c -> b (t n) c', b=b, t=t)
            img_encoded.append(self.image_encoders[view_idx](obs_lat)  # b t n c
                               # rearrange(
                               #     TensorUtils.time_distributed(
                               #         obs[:, view_idx, ...], self.image_encoders[view_idx]
                               #     ),
                               #     "b t c h w -> b t (h w) c",
                               # )
                               )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        return img_encoded

    def spatial_encode3(self, time, img_encoded, task_emb, extra_states, return_recon=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """

        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        # text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        # text_encoded = text_encoded.view(B, 1, -1)  # (b, 1, c)

        # 3. encode track
        # track_encoded, _recon_track = self.track_encode(track_obs, task_emb)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
        # # patch position embedding
        # tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
        # tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
        # # track id embedding
        # tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
        # track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        # track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            text_encoded += self.modality_embed[:, -1, :]
            img_track_text_encoded = torch.cat([img_encoded, text_encoded], -2)  # (b, 2*num_img_patch + 1, c)
            # img_track_text_encoded += self.modality_embed[:, self.modality_idx, :]
        else:
            # img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded = img_encoded
            # img_track_text_encoded += self.modality_embed[:, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 1, c)

        # 5. pass through transformer
        # encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        encoded = rearrange(encoded, "b t n c -> (b t) n c")
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t,  (b t) c
        out = self.spatial_downsample(out).reshape(B, T, 1, -1)
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        out_seq = [action_cls_token, out]
        # out = rearrange(out, '(b t) n c -> (b n) t c', b=B)
        # out = torch.cat((self.spatial_token.repeat(out.shape[0], 1, 1), out), dim=1)
        # out = self.time_img_transformer(out)[:, 0]
        # out = rearrange(out, '(b n) c -> b n c', b=B)
        # 6. encode extra states
        # for k in extra_states.keys():
        #     print(extra_states[k].shape)
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')
            # extra = extra.view(B, -1, extra.shape[-1])
            # extra = torch.cat([extra, extra[:, -1:].repeat(1, T-extra.shape[1], 1, 1)], dim=1)
            # extra = extra.expand(B, T, extra.shape[-2], extra.shape[-1])

        # 7. encode language, treat it as action token
        text_encoded_ = self.tokenizer(task_emb, padding="max_length", return_tensors="pt")['input_ids'].to(
            out.device)
        with torch.no_grad():
            text_encoded_ = self.siglip.get_text_features(text_encoded_)[1]
        text_encoded_ = self.language_encoder_temporal(text_encoded_)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, action_cls_token.shape[-1])  # (b, 1, c')
        # action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [out, text_encoded_]
        else:
            out_seq = [out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        # for _ in out_seq:
        #     print(_.shape)
        # out_seq.append(text_encoded_)
        time_emb = self.time_embed(time.expand(B)).unsqueeze(1).unsqueeze(1)  # b 1 1 c'
        time_emb = time_emb + text_encoded_
        out_seq.append(time_emb.repeat(1, T, 1, 1))
        # for tmp in out_seq:
        #     print(tmp.shape)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)

        return output


    def spatial_encode2(self, wm_act, time, obs, task_emb, extra_states, use_action, return_recon=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            obs_tmp = obs[:, view_idx, ...]  # b t c h w
            b, t = obs_tmp.shape[:2]
            with torch.no_grad():
                obs_lat = self.extract_latent(obs_tmp)  # b t n c
            cond = [obs_lat[:, -4:].clone(), None, wm_act]
            imagined_lat = self.track.sample_from_latent(cond=cond)
            tmp = obs_lat[:, -1:].clone().repeat(1, self.track.transformer.tube_size, 1, 1)
            imagined_lat = imagined_lat * use_action.reshape(use_action.shape[0], 1, 1, 1) + \
                           tmp * (1 - use_action).reshape(use_action.shape[0], 1, 1, 1)
            obs_lat = torch.cat([obs_lat[:, -self.max_seq_len:], imagined_lat], dim=1)  # b t n c
            # obs_lat = obs_lat.reshape(b, t, obs_lat.shape[-2], obs_lat.shape[-1])
            # obs_lat = rearrange(obs_lat, '(b t) n c -> b (t n) c', b=b, t=t)
            img_encoded.append(self.image_encoders[view_idx](obs_lat)  # b t n c
                               # rearrange(
                               #     TensorUtils.time_distributed(
                               #         obs[:, view_idx, ...], self.image_encoders[view_idx]
                               #     ),
                               #     "b t c h w -> b t (h w) c",
                               # )
                               )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        # text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        # text_encoded = text_encoded.view(B, 1, -1)  # (b, 1, c)

        # 3. encode track
        # track_encoded, _recon_track = self.track_encode(track_obs, task_emb)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
        # # patch position embedding
        # tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
        # tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
        # # track id embedding
        # tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
        # track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        # track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            text_encoded += self.modality_embed[:, -1, :]
            img_track_text_encoded = torch.cat([img_encoded, text_encoded], -2)  # (b, 2*num_img_patch + 1, c)
            # img_track_text_encoded += self.modality_embed[:, self.modality_idx, :]
        else:
            # img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded = img_encoded
            # img_track_text_encoded += self.modality_embed[:, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 1, c)

        # 5. pass through transformer
        # encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        encoded = rearrange(encoded, "b t n c -> (b t) n c")
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t,  (b t) c
        out = self.spatial_downsample(out).reshape(B, T, 1, -1)
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        out_seq = [action_cls_token, out]
        # out = rearrange(out, '(b t) n c -> (b n) t c', b=B)
        # out = torch.cat((self.spatial_token.repeat(out.shape[0], 1, 1), out), dim=1)
        # out = self.time_img_transformer(out)[:, 0]
        # out = rearrange(out, '(b n) c -> b n c', b=B)
        # 6. encode extra states
        # for k in extra_states.keys():
        #     print(extra_states[k].shape)
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')
            # extra = extra.view(B, -1, extra.shape[-1])
            # extra = torch.cat([extra, extra[:, -1:].repeat(1, T-extra.shape[1], 1, 1)], dim=1)
            # extra = extra.expand(B, T, extra.shape[-2], extra.shape[-1])

        # 7. encode language, treat it as action token
        text_encoded_ = self.tokenizer(task_emb, padding="max_length", return_tensors="pt")['input_ids'].to(
            out.device)
        with torch.no_grad():
            text_encoded_ = self.siglip.get_text_features(text_encoded_)[1]
        text_encoded_ = self.language_encoder_temporal(text_encoded_)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, action_cls_token.shape[-1])  # (b, 1, c')
        # action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [out, text_encoded_]
        else:
            out_seq = [out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        # for _ in out_seq:
        #     print(_.shape)
        # out_seq.append(text_encoded_)
        time_emb = self.time_embed(time.expand(B)).unsqueeze(1).unsqueeze(1)  # b 1 1 c'
        time_emb = time_emb + text_encoded_
        out_seq.append(time_emb.repeat(1, T, 1, 1))
        # for tmp in out_seq:
        #     print(tmp.shape)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)

        return output

    def spatial_encode(self, obs, track_obs, task_emb, extra_states, return_recon=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)

        # 3. encode track
        # track_encoded, _recon_track = self.track_encode(track_obs, task_emb)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
        # # patch position embedding
        # tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
        # tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
        # # track id embedding
        # tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
        # track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        # track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            img_track_text_encoded = torch.cat([img_encoded, text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 1, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
        else:
            # img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded = img_encoded
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)

        # 5. pass through transformer
        encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 6. encode extra states
        # for k in extra_states.keys():
        #     print(extra_states[k].shape)
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')
            extra = torch.cat([extra, extra[:, -1:].repeat(1, T-extra.shape[1], 1, 1)], dim=1)
            # extra = extra.expand(B, T, extra.shape[-2], extra.shape[-1])

        # 7. encode language, treat it as action token
        text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        # for _ in out_seq:
        #     print(_.shape)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)

        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        # return x  # (b, t, c)
        return x[:, :, 0]  # (b, t, c)


    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + time * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - time * noise
        return x0

    def sample_act(self, obs, task_emb, extra_states, wm_act, use_action):
        batch, device, sampling_timesteps = obs.shape[0], obs.device, self.sampling_step
        # get condition features
        img_encoded = self.image_encode(obs, wm_act, use_action)
        if 'ee_states' in extra_states:
            ee_states = extra_states['ee_states']
            ee_states = self.apply_act(ee_states, wm_act)
            extra_states['ee_states'] = ee_states
        if 'joint_states' in extra_states:
            joint_states = extra_states['joint_states']
            joint_states = self.apply_act2(joint_states, wm_act)
            extra_states['joint_states'] = joint_states
        if 'gripper_states' in extra_states:
            gripper_states = extra_states['gripper_states']
            gripper_states = self.apply_act2(gripper_states, wm_act)
            extra_states['gripper_states'] = gripper_states
        shape = (batch, *self.act_shape)
        step = 1. / sampling_timesteps
        rho = 1.
        sigma_max = 1.
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float32, device=device)
        # t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
        #             self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        if sampling_timesteps > 1:
            t_steps = (sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
                    step - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        else:
            t_steps = torch.tensor([1., 0.], device=device)
        alpha = 1
        x_next = torch.randn(shape, device=device, dtype=torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            pred = self.forward_sample2(x_cur, t_cur, img_encoded, task_emb, extra_states)
            C, noise = pred[:2]
            C, noise = C.to(torch.float32), noise.to(torch.float32)
            x0 = x_cur - C * t_cur - noise * t_cur
            x0 = torch.clamp(x0, -1, 1)
            # d_cur = (x_cur - x0) / t_cur
            # x_next = x_cur + (t_next - t_cur) * d_cur
            x_next = x0 + t_next * C + t_next * noise
        action = x_next.to(torch.float32)
        return action

    def apply_act(self , ee_states, act):
        ee_pos = ee_states[:, :, :3].clone()
        ee_ori = ee_states[:, :, 3:6].clone()
        cur_pos, cur_ori = ee_pos[:, -1], ee_ori[:, -1]
        act_pos = act[:, :, :3]
        act_ori = act[:, :, 3:6]
        act_gri = act[:, :, 6:]
        pred_pos, pred_ori = [], []
        for i in range(act.shape[1]):
            new_pos = cur_pos + act_pos[:, i] * 0.05
            pred_pos.append(new_pos.clone())
            cur_pos = new_pos
            cur_quat = axisangle2quat_torch(cur_ori)
            cur_rot = quat2mat_torch(cur_quat)
            del_rot = quat2mat_torch(axisangle2quat_torch(act_ori[:, i] * 0.5))
            new_rot = torch.bmm(del_rot, cur_rot)
            new_quat = mat2quat_torch(new_rot)
            new_ori = quat2axisangle_torch(new_quat)
            pred_ori.append(new_ori.clone())
            cur_ori = new_ori
        pred_pos = torch.stack(pred_pos, dim=1)
        pred_ori = torch.stack(pred_ori, dim=1)
        if ee_states.shape[-1] == 7:
            pred_ee_states = torch.cat([pred_pos, pred_ori, act_gri], dim=-1)
        else:
            pred_ee_states = torch.cat([pred_pos, pred_ori], dim=-1)
        ee_states = torch.cat([ee_states, pred_ee_states], dim=1)
        return ee_states

    def apply_act2(self , ee_states, act):
        ee_latest = ee_states[:, -1].clone()
        future_frames = act.shape[1]
        # use latest pose
        pred_ee_states = ee_latest.unsqueeze(1).repeat(1, future_frames, 1)
        ee_states = torch.cat([ee_states, pred_ee_states], dim=1)
        return ee_states

    def forward(self, action, obs, track, task_emb, extra_states, use_action, mode='train'):
        """
        Return feature and info.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            extra_states: {k: b t e}
        """
        time = torch.rand(action.shape[0], device=action.device) * (1. - 0.001) + 0.001  # diffusion time step
        b = obs.shape[0]
        # obs_view1 = obs[:, 0]
        # obs_view2 = obs[:, 1]
        # with torch.no_grad():
        #     cond1 = [obs_view1[:, -1], task_emb]
        #     pred_view1 = self.track.sample(batch_size=b, cond=cond1)  # b, t, c, h, w
        #     cond2 = [obs_view2[:, -1], task_emb]
        #     pred_view2 = self.track.sample(batch_size=b, cond=cond2)
        # obs_view1 = torch.cat([obs_view1, pred_view1], dim=1)
        # obs_view2 = torch.cat([obs_view2, pred_view2], dim=1)
        # obs = torch.stack([obs_view1, obs_view2], dim=1)  # b v t c h w
        # obs = F.interpolate(obs.flatten(1, 3), size=(126, 126), mode="bilinear")
        # obs = rearrange(obs, 'b (v t c) h w -> b v t c h w', v=2, c=3)
        # obs = obs[:, :, -10:]
        wm_act = action[:, :-1].clone()
        if random.random() >= 0.5:
            wm_act_mean = wm_act.mean(dim=1, keepdims=True).repeat(1, wm_act.shape[1], 1)
            wm_act = wm_act + torch.randn_like(wm_act) * 0.05 * wm_act_mean
        wm_act = wm_act.clamp(-1, 1)
        if 'ee_states' in extra_states:
            ee_states = extra_states['ee_states']
            ee_states = self.apply_act(ee_states, wm_act)
            extra_states['ee_states'] = ee_states
        if 'joint_states' in extra_states:
            joint_states = extra_states['joint_states']
            joint_states = self.apply_act2(joint_states, wm_act)
            extra_states['joint_states'] = joint_states
        if 'gripper_states' in extra_states:
            gripper_states = extra_states['gripper_states']
            gripper_states = self.apply_act2(gripper_states, wm_act)
            extra_states['gripper_states'] = gripper_states
        x = self.spatial_encode2(wm_act, time, obs, task_emb, extra_states, use_action, return_recon=False)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)
        # t = x.shape[1]

        # recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        # x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)

        noise = torch.randn_like(action)
        C = - action
        noisy_act = self.q_sample(action, noise, time, C)
        input_emb = self.input_proj_act(noisy_act)
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pe_query[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        q = token_embeddings + position_embeddings
        # (B,T,n_emb)
        q = self.decoder(
            tgt=q,
            memory=x,
            tgt_mask=None,
            memory_mask=None
        )
        # dist = self.policy_head(x)  # only use the current timestep feature to predict action
        # dist = dist.reshape(b, *self.act_shape)
        C_pred = self.policy_head1(q)
        noise_pred = self.policy_head2(q)
        if mode=='train':
            loss = F.mse_loss(C_pred, C, reduction="mean") + F.mse_loss(noise_pred, noise, reduction="mean")
            return loss
        elif mode=='sample':
            return C_pred, noise_pred

    def forward_sample(self, x_cur, t_cur, obs, task_emb, extra_states, wm_act, use_action):
        # b = obs.shape[0]
        # obs_view1 = obs[:, 0]
        # obs_view2 = obs[:, 1]
        # with torch.no_grad():
        #     cond1 = [obs_view1[:, -1], task_emb]
        #     pred_view1 = self.track.sample(batch_size=b, cond=cond1)  # b, t, c, h, w
        #     cond2 = [obs_view2[:, -1], task_emb]
        #     pred_view2 = self.track.sample(batch_size=b, cond=cond2)
        # obs_view1 = torch.cat([obs_view1, pred_view1], dim=1)
        # obs_view2 = torch.cat([obs_view2, pred_view2], dim=1)
        # obs = torch.stack([obs_view1, obs_view2], dim=1)  # b v t c h w
        # obs = obs[:, :, -10:]
        x = self.spatial_encode2(wm_act, t_cur, obs, task_emb, extra_states, use_action,
                                 return_recon=False)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)
        # t = x.shape[1]

        # recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        # x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)
        input_emb = self.input_proj_act(x_cur)
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pe_query[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        q = token_embeddings + position_embeddings
        # (B,T,n_emb)
        q = self.decoder(
            tgt=q,
            memory=x,
            tgt_mask=None,
            memory_mask=None
        )
        # dist = self.policy_head(x)  # only use the current timestep feature to predict action
        # dist = dist.reshape(b, *self.act_shape)
        C_pred = self.policy_head1(q)
        noise_pred = self.policy_head2(q)
        return C_pred, noise_pred

    def forward_sample2(self, x_cur, t_cur, img_encoded, task_emb, extra_states):
        # b = obs.shape[0]
        # obs_view1 = obs[:, 0]
        # obs_view2 = obs[:, 1]
        # with torch.no_grad():
        #     cond1 = [obs_view1[:, -1], task_emb]
        #     pred_view1 = self.track.sample(batch_size=b, cond=cond1)  # b, t, c, h, w
        #     cond2 = [obs_view2[:, -1], task_emb]
        #     pred_view2 = self.track.sample(batch_size=b, cond=cond2)
        # obs_view1 = torch.cat([obs_view1, pred_view1], dim=1)
        # obs_view2 = torch.cat([obs_view2, pred_view2], dim=1)
        # obs = torch.stack([obs_view1, obs_view2], dim=1)  # b v t c h w
        # obs = obs[:, :, -10:]
        x = self.spatial_encode3(t_cur, img_encoded, task_emb, extra_states,
                                 return_recon=False)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)
        # t = x.shape[1]

        # recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        # x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)
        input_emb = self.input_proj_act(x_cur)
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pe_query[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        q = token_embeddings + position_embeddings
        # (B,T,n_emb)
        q = self.decoder(
            tgt=q,
            memory=x,
            tgt_mask=None,
            memory_mask=None
        )
        # dist = self.policy_head(x)  # only use the current timestep feature to predict action
        # dist = dist.reshape(b, *self.act_shape)
        C_pred = self.policy_head1(q)
        noise_pred = self.policy_head2(q)
        return C_pred, noise_pred

    def forward_loss(self, obs, track, task_emb, extra_states, action, use_action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            task_emb: b emb_size
            action: b t act_dim
        """
        obs, track, action = self.preprocess(obs, track, action)

        loss = self.forward(action, obs, track, task_emb, extra_states, use_action)
        # loss = self.policy_head.loss_fn(dist, action, reduction="mean")

        ret_dict = {
            "bc_loss": loss.sum().item(),
        }

        # if not self.policy_head.deterministic:
        #     # pseudo loss
        #     sampled_action = dist.sample().detach()
        #     mse_loss = F.mse_loss(sampled_action, action)
        #     ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

        ret_dict["loss"] = ret_dict["bc_loss"]
        return loss.sum(), ret_dict

    def forward_vis(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2
            task_emb: b emb_size
        Returns:
        """
        _, track, _ = self.preprocess(obs, track, action)
        track = track[:, :, 0, :, :, :]  # (b, v, track_len, n, 2) use the track in the first timestep

        b, v, t, track_obs_t, c, h, w = track_obs.shape
        if t >= self.num_track_ts:
            track_obs = track_obs[:, :, :self.num_track_ts, ...]
            track = track[:, :, :self.num_track_ts, ...]
        else:
            last_obs = track_obs[:, :, -1:, ...]
            pad_obs = repeat(last_obs, "b v 1 track_obs_t c h w -> b v t track_obs_t c h w", t=self.num_track_ts-t)
            track_obs = torch.cat([track_obs, pad_obs], dim=2)
            last_track = track[:, :, -1:, ...]
            pad_track = repeat(last_track, "b v 1 n d -> b v tl n d", tl=self.num_track_ts-t)
            track = torch.cat([track, pad_track], dim=2)

        grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
        grid_track = repeat(grid_points, "n d -> b v tl n d", b=b, v=v, tl=self.num_track_ts)

        all_ret_dict = {}
        for view in range(self.num_views):
            gt_track = track[:1, view]  # (1 tl n d)
            gt_track_vid = tracks_to_video(gt_track, img_size=h)
            combined_gt_track_vid = (track_obs[:1, view, 0, :, ...] * .25 + gt_track_vid * .75).cpu().numpy().astype(np.uint8)

            _, ret_dict = self.track.forward_vis(track_obs[:1, view, 0, :, ...], grid_track[:1, view], task_emb[:1], p_img=0)
            ret_dict["combined_track_vid"] = np.concatenate([combined_gt_track_vid, ret_dict["combined_track_vid"]], axis=-1)

            all_ret_dict = {k: all_ret_dict.get(k, []) + [v] for k, v in ret_dict.items()}

        for k, v in all_ret_dict.items():
            if k == "combined_image" or k == "combined_track_vid":
                all_ret_dict[k] = np.concatenate(v, axis=-2)  # concat on the height dimension
            else:
                all_ret_dict[k] = np.mean(v)
        return None, all_ret_dict

    def act(self, obs, task_emb, extra_states, wm_act, use_action):
        """
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """
        self.eval()
        B = obs.shape[0]

        # expand time dimenstion
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
        extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        use_action = use_action.to(device)
        wm_act = wm_act.to(device)
        # task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}
        if 'ee_states' in extra_states:
            ee_states = extra_states['ee_states']
            ee_ori = ee_states[:, 0, 3:6]
            ee_quat = axisangle2quat_torch(ee_ori)
            negetive_mask = ee_quat[:, -1] < 0
            ee_quat[negetive_mask] = torch.negative(ee_quat[negetive_mask])
            ee_ori = quat2axisangle_torch(ee_quat)
            ee_states[:, 0, 3:6] = ee_ori
            extra_states['ee_states'] = ee_states

        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)

        while len(self.track_obs_queue) < self.max_seq_len-1:
            # self.track_obs_queue.append(torch.zeros_like(obs))
            self.track_obs_queue.append(obs.clone())
            if 'joint_states' in self.extra_state_keys:
                self.joint_state_queue.append(extra_states['joint_states'].clone())
            if 'gripper_states' in self.extra_state_keys:
                self.gripper_state_queue.append(extra_states['gripper_states'].clone())
            if 'ee_pos' in self.extra_state_keys:
                self.ee_pos_queue.append(extra_states['ee_pos'].clone())
            if 'ee_states' in self.extra_state_keys:
                self.ee_state_queue.append(extra_states['ee_states'].clone())

        self.track_obs_queue.append(obs.clone())
        if 'joint_states' in self.extra_state_keys:
            self.joint_state_queue.append(extra_states['joint_states'].clone())
        if 'gripper_states' in self.extra_state_keys:
            self.gripper_state_queue.append(extra_states['gripper_states'].clone())
        if 'ee_pos' in self.extra_state_keys:
            self.ee_pos_queue.append(extra_states['ee_pos'].clone())
        if 'ee_states' in self.extra_state_keys:
            self.ee_state_queue.append(extra_states['ee_states'].clone())
        # track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        # track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")
        track_obs = None
        extra_states_dict = {}
        if 'joint_states' in self.extra_state_keys:
            extra_states_dict.update({'joint_states': torch.cat(list(self.joint_state_queue), dim=1)})
        if 'gripper_states' in self.extra_state_keys:
            extra_states_dict.update({'gripper_states': torch.cat(list(self.gripper_state_queue), dim=1)})
        if 'ee_pos' in self.extra_state_keys:
            extra_states_dict.update({'ee_pos': torch.cat(list(self.ee_pos_queue), dim=1)})
        if 'ee_states' in self.extra_state_keys:
            extra_states_dict.update({'ee_states': torch.cat(list(self.ee_state_queue), dim=1)})
        # extra_states = {
        #     'joint_states': torch.cat(list(self.joint_state_queue), dim=1),
        #     'gripper_states': torch.cat(list(self.gripper_state_queue), dim=1),
        # }

        obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v t c h w
        obs = self._preprocess_rgb(obs)
        b = obs.shape[0]
        # obs_view1 = obs[:, 0]
        # obs_view2 = obs[:, 1]
        # with torch.no_grad():
        #     cond1 = [obs_view1[:, -1], task_emb]
        #     pred_view1 = self.track.sample(batch_size=b, cond=cond1)  # b, t, c, h, w
        #     cond2 = [obs_view2[:, -1], task_emb]
        #     pred_view2 = self.track.sample(batch_size=b, cond=cond2)
        # obs_view1 = torch.cat([obs_view1, pred_view1], dim=1)  # b t c h w
        # obs_view2 = torch.cat([obs_view2, pred_view2], dim=1)  # b t c h w
        # obs = torch.stack([obs_view1, obs_view2], dim=1)  # b v t c h w
        # pred_view = torch.stack([pred_view1, pred_view2], dim=1)  # b v t c h w
        # obs = F.interpolate(obs.flatten(1, 3), size=(126, 126), mode="bilinear")
        # obs = rearrange(obs, 'b (v t c) h w -> b v t c h w', v=2, c=3)
        # obs = obs[:, :, -10:]
        with torch.no_grad():
            action = self.sample_act(obs, task_emb=task_emb, extra_states=extra_states_dict, wm_act=wm_act, use_action=use_action)
            # x = self.spatial_encode(obs, track_obs, task_emb=task_emb, extra_states=extra_states, return_recon=False)  # x: (b, 1, 4, c), recon_track: (b, v, 1, tl, n, 2)
            # self.latent_queue.append(x)
            # x = torch.cat(list(self.latent_queue), dim=1)  # (b, t, 4, c)
            # x = self.temporal_encode(x)  # (b, t, c)
            # x = x.flatten(1, 2)  # b t*c
            # feat = torch.cat([x[:, -1], rearrange(rec_tracks[:, :, -1, :, :, :], "b v tl n d -> b (v tl n d)")], dim=-1)
            # action = self.policy_head(x)  # only use the current timestep feature to predict action
            action = action.detach().cpu()  # (b, act_dim)

        action = action.reshape(-1, *self.act_shape)
        action = torch.clamp(action, -1, 1)
        action_np = action.float().cpu().numpy()[:, 0]
        return action_np, obs, action[:, 1:]
        # return action.float().cpu().numpy(), (None, pred_view)  # (b, *act_shape)

    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()
        self.joint_state_queue.clear()
        self.gripper_state_queue.clear()
        self.ee_pos_queue.clear()
        self.ee_state_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def train(self, mode=True):
        super().train(mode)
        # self.track.eval()

    def eval(self):
        super().eval()
        # self.track.eval()
