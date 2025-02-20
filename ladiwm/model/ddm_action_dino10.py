import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from torch import nn
import math
from ladiwm.utils.flow_utils import ImageUnNormalize, tracks_to_video
from ladiwm.utils.pos_embed_utils import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from ladiwm.policy.vilt_modules.language_modules import *
from .track_patch_embed import TrackPatchEmbed
from .transformer import Transformer
from ladiwm.utils.build_utils import construct_class_by_name

# use historical latent feature instead of image as condition
# dual diffusion and interaction
# torch.diff() to calculate residual

class DDM(nn.Module):
    """
    flow video model using a BERT transformer

    dim: int, dimension of the model
    depth: int, number of layers
    heads: int, number of heads
    dim_head: int, dimension of each head
    attn_dropout: float, dropout for attention layers
    ff_dropout: float, dropout for feedforward layers
    """

    def __init__(self,
                 transformer_cfg,
                 track_cfg,
                 vid_cfg,
                 language_encoder_cfg,
                 vae,
                 transformer,
                 seq_len=197,
                 channels=32,
                 pred_flow=False,
                 pred_mask=False,
                 load_path=None):
        super().__init__()
        self.dim = 192
        self.dino_scale = 1.
        self.siglip_scale = 1.
        self.seq_len = seq_len
        self.transformer = transformer
        self.vae = vae
        for n, p in self.vae.named_parameters():
            p.requires_grad = False
            # if 'encoder' in n:
            #     p.requires_grad = False
        self.pred_flow = pred_flow
        self.pred_mask = pred_mask
        # if pred_flow:
        #     self.flow_layer = FlowNet(4, 2, 64)

        # if pred_mask:
        #     self.seg_layer = SegNet(4, 2, 64)
        # self.track_proj_encoder, self.track_decoder = self._init_track_modules(**track_cfg, dim=dim)
        # self.img_proj_encoder, self.img_decoder = self._init_video_modules(**vid_cfg, dim=dim)
        # self.language_encoder = self._init_language_encoder(output_size=dim, **language_encoder_cfg)
        # self._init_weights(self.dim, self.num_img_patches)
        self.sampling_timesteps = 10
        self.mask_token = nn.Parameter(torch.randn(1, 81, 12))
        self.channels = channels
        # self.self_condition = self.model.self_condition
        self.register_buffer('eps', torch.tensor(1e-3))
        # self.sigma_min = cfg.get('sigma_min', 1e-2) if cfg is not None else 1e-2
        # self.sigma_max = cfg.get('sigma_max', 1) if cfg is not None else 1
        self.weighting_loss = True
        if self.weighting_loss:
            print('#### WEIGHTING LOSS ####')

        # self.clip_x_start = clip_x_start
        self.image_size = [128, 128]

        if load_path is not None:
            self.load(load_path)
            print(f"loaded model from {load_path}")

    # def _init_transformer(self, dim, dim_head, heads, depth, attn_dropout, ff_dropout):
    #     self.transformer = Transformer(
    #         dim=dim,
    #         dim_head=dim_head,
    #         heads=heads,
    #         depth=depth,
    #         attn_dropout=attn_dropout,
    #         ff_dropout=ff_dropout)
    #
    #     return self.transformer

    # def _init_track_modules(self, dim, num_track_ts, num_track_ids, patch_size=1):
    #     self.num_track_ts = num_track_ts
    #     self.num_track_ids = num_track_ids
    #     self.track_patch_size = patch_size
    #
    #     self.track_proj_encoder = TrackPatchEmbed(
    #         num_track_ts=num_track_ts,
    #         num_track_ids=num_track_ids,
    #         patch_size=patch_size,
    #         in_dim=2,
    #         embed_dim=dim)
    #     self.num_track_patches = self.track_proj_encoder.num_patches
    #     self.track_decoder = nn.Linear(dim, 2 * patch_size, bias=True)
    #     self.num_track_ids = num_track_ids
    #     self.num_track_ts = num_track_ts
    #
    #     return self.track_proj_encoder, self.track_decoder
    #
    # def _init_video_modules(self, dim, img_size, patch_size, frame_stack=1, img_mean=[.5, .5, .5], img_std=[.5, .5, .5]):
    #     self.img_normalizer = T.Normalize(img_mean, img_std)
    #     self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
    #     if isinstance(img_size, int):
    #         img_size = (img_size, img_size)
    #     else:
    #         img_size = (img_size[0], img_size[1])
    #     self.img_size = img_size
    #     self.frame_stack = frame_stack
    #     self.patch_size = patch_size
    #     self.img_proj_encoder = PatchEmbed(
    #         img_size=img_size,
    #         patch_size=patch_size,
    #         in_chans=3 * self.frame_stack,
    #         embed_dim=dim,
    #     )
    #     self.num_img_patches = self.img_proj_encoder.num_patches
    #     self.img_decoder = nn.Linear(dim, 3 * self.frame_stack * patch_size ** 2, bias=True)
    #
    #     return self.img_proj_encoder, self.img_decoder

    def _init_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _init_weights(self, dim, num_img_patches):
        """
        initialize weights; freeze all positional embeddings
        """
        num_track_t = self.num_track_ts // self.track_patch_size

        self.track_embed = nn.Parameter(torch.randn(1, num_track_t, 1, dim), requires_grad=True)
        self.img_embed = nn.Parameter(torch.randn(1, num_img_patches, dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))

        track_embed = get_1d_sincos_pos_embed(dim, num_track_t)
        track_embed = rearrange(track_embed, 't d -> () t () d')
        self.track_embed.data.copy_(torch.from_numpy(track_embed))

        num_patches_h, num_patches_w = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        img_embed = get_2d_sincos_pos_embed(dim, (num_patches_h, num_patches_w))
        img_embed = rearrange(img_embed, 'n d -> () n d')
        self.img_embed.data.copy_(torch.from_numpy(img_embed))

        print(f"num_track_patches: {self.num_track_patches}, num_img_patches: {num_img_patches}, total: {self.num_track_patches + num_img_patches}")

    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + time * noise
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - time * noise
        return x0

    def _preprocess_track(self, track):
        return track

    def _preprocess_vis(self, vis):
        return vis

    def _preprocess_vid(self, vid):
        assert torch.max(vid) >= 2

        vid = vid[:, -self.frame_stack:]
        vid = self.img_normalizer(vid / 255.)
        return vid

    def _encode_track(self, track):
        """
        track: (b, t, n, 2)
        """
        b, t, n, _ = track.shape
        track = self._mask_track_as_first(track)  # b, t, n, d. track embedding is 1, t, 1, d
        track = self.track_proj_encoder(track)

        track = track + self.track_embed
        track = rearrange(track, 'b t n d -> b (t n) d')
        return track

    def _encode_video(self, vid, p):
        """
        vid: (b, t, c, h, w)
        """
        vid = rearrange(vid, "b t c h w -> b (t c) h w")
        patches = self.img_proj_encoder(vid)  # b, n, d
        patches = self._mask_patches(patches, p=p)
        patches = patches + self.img_embed

        return patches

    def _mask_patches(self, patches, p):
        """
        mask patches according to p
        """
        b, n, _ = patches.shape
        mask = torch.rand(b, n, device=patches.device) < p
        masked_patches = patches.clone()
        masked_patches[mask] = self.mask_token
        return masked_patches

    def _mask_track_as_first(self, track):
        """
        mask out all frames to have the same token as the first frame
        """
        mask_track = track.clone() # b, t, n, d
        mask_track[:, 1:] = track[:, [0]]
        return mask_track

    def forward(self, vid, track, task_emb, p_img):
        """
        track: (b, tl, n, 2), which means current time step t0 -> t0 + tl
        vid: (b, t, c, h, w), which means the past time step t0 - t -> t0
        task_emb, (b, emb_size)
        """
        assert torch.max(vid) <=1.
        B, T, _, _ = track.shape
        patches = self._encode_video(vid, p_img)  # (b, n_image, d)
        enc_track = self._encode_track(track)

        text_encoded = self.language_encoder(task_emb)  # (b, c)
        text_encoded = rearrange(text_encoded, 'b c -> b 1 c')

        x = torch.cat([enc_track, patches, text_encoded], dim=1)
        x = self.transformer(x)

        rec_track, rec_patches = x[:, :self.num_track_patches], x[:, self.num_track_patches:-1]
        rec_patches = self.img_decoder(rec_patches)  # (b, n_image, 3 * t * patch_size ** 2)
        rec_track = self.track_decoder(rec_track)  # (b, (t n), 2 * patch_size)
        num_track_h = self.num_track_ts // self.track_patch_size
        rec_track = rearrange(rec_track, 'b (t n) (p c) -> b (t p) n c', p=self.track_patch_size, t=num_track_h)

        return rec_track, rec_patches

    def reconstruct(self, vid, track, task_emb, p_img):
        """
        wrapper of forward with preprocessing
        track: (b, tl, n, 2), which means current time step t0 -> t0 + tl
        vid: (b, t, c, h, w), which means the past time step t0 - t -> t0
        task_emb: (b, e)
        """
        assert len(vid.shape) == 5  # b, t, c, h, w
        track = self._preprocess_track(track)
        vid = self._preprocess_vid(vid)
        return self.forward(vid, track, task_emb, p_img)

    def normalize(self, x):  # [0, 255] --> [-1, 1]
        x = x / 255.
        # x = 2 * x - 1.
        return x

    def unnormalize(self, x):  # [-1, 1] --> [0, 255]
        # x = (x + 1.) / 2
        x = x.clamp(0., 1.)
        x = x * 255.
        return x

    def forward_loss(self,
                     his,
                     preds,
                     track,
                     task_emb,
                     action,
                     lbd_track,
                     lbd_img,
                     p_img,
                     return_outs=False,
                     vis=None, **kwargs):
        """
        track: (b, tl, n, 2), which means current time step t0 -> t0 + tl
        vid: (b, t, c, h, w), which means the past time step t0 - t -> t0
        task_emb: (b, e)
        """
        B, T1, C, H, W = his.shape
        his = self.normalize(his)  # [-1, 1]
        # img_cur = his[:, -1]  # b c h w
        # img_cur = self.normalize(img_cur)  # [-1, 1]
        # img_mask = img_cur.clone()
        with torch.no_grad():
            lc_his = self.vae.encode5(his.flatten(0, 1))
            lc_his = lc_his.detach().reshape(B, T1, lc_his.shape[-2], lc_his.shape[-1])  # b*t, n, c
            lc_his_dino, lc_his_sig = lc_his.chunk(2, dim=2)
            # lc_cur = self.vae.encode4(img_cur)#.sample()  # b n c
        x_prior_dino = lc_his_dino[:, -1].clone().unsqueeze(1)  # b 1 n c
        x_prior_sig = lc_his_sig[:, -1].clone().unsqueeze(1)  # b 1 n c
        # lc_his = lc_his[:, :-1]  # b t n c
        # x_prior = x_prior * 0.18215
        # mae training
        '''
        mask_ = torch.rand_like(lc_cur)
        mask_ = mask_ >= 0.5
        mask_[:(B // 2), :] = False
        mask_ = mask_.float()
        # x_mask = x.clone() * mask_
        lc_cur_mask = lc_cur.clone()
        lc_cur_mask = lc_cur_mask * (1 - mask_) + self.mask_token.repeat(B, 1, 1) * mask_
        rec_cur = self.vae.decode(lc_cur_mask)  # [-1, 1]
        '''

        img_next = preds  # b t c h w
        T2 = preds.shape[1]
        img_next = self.normalize(img_next)  # [-1, 1]
        with torch.no_grad():
            lc_next = self.vae.encode5(img_next.flatten(0, 1))
            lc_next = lc_next.detach()
        N, C = lc_next.shape[-2:]
        # z2 = z2.reshape(B, T, C, H, W).flatten(1, 2)  # B, T*4, H, W
        lc_next = lc_next.reshape(B, T2, N, C)#.transpose(1, 2)  # B, 4, T, H, W
        lc_next_dino, lc_next_sig = lc_next.chunk(2, dim=2)

        # x_start = lc_next #* 0.18215
        b, tl, n, _ = track.shape
        # if vis is None:
        #     vis = torch.ones((b, tl, n)).to(track.device)
        #
        # track = self._preprocess_track(track)
        # vid = self._preprocess_vid(vid)
        # vis = self._preprocess_vis(vis)

        # res_dino = (lc_next_dino - x_prior_dino) * self.dino_scale
        # res_sig = (lc_next_sig - x_prior_sig) * self.siglip_scale
        res_dino = torch.cat([x_prior_dino, lc_next_dino], dim=1)
        res_dino = torch.diff(res_dino, dim=1) * self.dino_scale
        res_sig = torch.cat([x_prior_sig, lc_next_sig], dim=1)
        res_sig = torch.diff(res_sig, dim=1) * self.siglip_scale
        # res = x_start - x_prior[:, :, -1].unsqueeze(2)
        C_dino = -1 * res_dino   # U(t) = Ct, U(1) = -x0
        C_sig = -1 * res_sig
        t = torch.rand(res_dino.shape[0], device=res_dino.device) * (1. - self.eps) + self.eps
        noise_dino = torch.randn_like(res_dino)
        x_noisy_dino = self.q_sample(x_start=res_dino, noise=noise_dino, t=t, C=C_dino)
        noise_sig = torch.randn_like(res_sig)
        x_noisy_sig = self.q_sample(x_start=res_sig, noise=noise_sig, t=t, C=C_sig)
        # C_pred, noise_pred = self.forward(vid, track, task_emb, p_img)
        cond = [lc_his_dino, lc_his_sig, task_emb, action]
        C_pred_dino, noise_pred_dino, C_pred_sig, noise_pred_sig, C_pred_dino_tmp, C_pred_sig_tmp = \
                                self.transformer(x_noisy_dino, x_noisy_sig, cond, t)
        # vis[vis == 0] = .1
        # vis = repeat(vis, "b tl n -> b tl n c", c=2)

        # simple_weight1 = ((t - 1) / t) ** 2 + 1
        # simple_weight2 = (t / (1 - t + self.eps)) ** 2 + 1
        simple_weight1 = 1
        simple_weight2 = 1
        rec_weight = -torch.log(t)
        C_loss = simple_weight1 * ((C_pred_dino - C_dino) ** 2).mean([1, 2, 3]) + \
                 simple_weight1 * ((C_pred_sig - C_sig) ** 2).mean([1, 2, 3])

        noise_loss = simple_weight2 * ((noise_pred_dino - noise_dino) ** 2).mean([1, 2, 3]) + \
                     simple_weight2 * ((noise_pred_sig - noise_sig) ** 2).mean([1, 2, 3])
        # track_loss = simple_weight1 * ((C_pred_dino - C_dino) ** 2).mean([1, 2, 3]) + \
        #              simple_weight2 * ((noise_pred_dino - noise_dino) ** 2).mean([1, 2, 3]) + \
        #              simple_weight1 * ((C_pred_sig - C_sig) ** 2).mean([1, 2, 3]) + \
        #              simple_weight2 * ((noise_pred_sig - noise_sig) ** 2).mean([1, 2, 3])
        track_loss = C_loss + noise_loss
        track_loss = track_loss.mean()
        aux_loss = rec_weight* ((C_pred_dino_tmp - C_dino) ** 2).mean([1, 2, 3]) \
                     + rec_weight * ((C_pred_sig_tmp - C_sig) ** 2).mean([1, 2, 3])
        aux_loss = aux_loss.mean()
        # if kwargs.pop('epoch') > 50:
        #     img_loss = torch.zeros_like(track_loss)
        variation_loss = (C_pred_sig.abs().mean([1, 2, 3]) + C_pred_dino.abs().mean([1, 2, 3])) * 0.001
        variation_loss = rec_weight * variation_loss
        variation_loss = variation_loss.mean()
        loss = track_loss + variation_loss + aux_loss

        ret_dict = {
            "loss": loss.item(),
            "track_loss": track_loss.item(),
            "aux_loss": aux_loss.item(),
            "variation_loss": variation_loss.item(),
            'C_loss': C_loss.mean().item(),
            'noise_loss': noise_loss.mean().item(),
        }
        # if self.pred_flow:
        #     x_rec = x_prior - C_pred * math.sqrt(2)
        #     rec_weight = -torch.log(t.reshape(C.shape[0], 1)) / 4
        #     flow_pred = self.flow_layer(x_rec, t, task_emb)
        #     flow = kwargs['flow']
        #     flow = F.interpolate(flow, size=flow_pred.shape[-2:], mode="bilinear")
        #     loss_flow = rec_weight * (flow_pred - flow).abs().sum([1, 2, 3])
        #     loss_flow = loss_flow.mean()
        #     loss += loss_flow
        #     ret_dict.update({f'loss_flow': loss_flow})
        # if self.pred_mask:
        #     x_rec = x_prior - C_pred * math.sqrt(2)
        #     rec_weight = -torch.log(t.reshape(C.shape[0], 1)) / 4
        #     mask_pred = self.seg_layer(x_rec, t, task_emb)
        #     mask = kwargs['mask']
        #     mask = F.interpolate(mask, size=mask_pred.shape[-2:], mode="bilinear")
        #     loss_mask = rec_weight * (mask_pred - mask).abs().sum([1, 2, 3])
        #     loss_mask = loss_mask.mean()
        #     loss += loss_mask
        #     ret_dict.update({f'loss_mask': loss_mask})
        if return_outs:
            return loss.sum(), ret_dict, (rec_track, rec_patches)
        return loss.sum(), ret_dict

    def evaluate_loss(self,
                      his,
                      preds,
                      track,
                      task_emb,
                      action,
                     lbd_track,
                     lbd_img,
                     p_img,
                     return_outs=False,
                     vis=None):
        B, T, C, H, W = his.shape
        his = self.normalize(his)  # [-1, 1]
        img_cur = his[:, -1]  # b c h w
        # img_cur = self.normalize(img_cur)  # [-1, 1]
        # lc_cur = self.vae.encode(img_cur)  # b c h w
        # rec_cur = self.vae.decode(lc_cur)
        # img_mask = img_cur.clone()
        # lc_cur = self.vae.encode(img_cur).sample()  # b c h w
        # x_prior = lc_cur.detach().clone().unsqueeze(2)  # b c 1 h w
        # x_prior = x_prior * 0.18215
        # masks = masks / 255.
        cond = [his, task_emb, action]
        rec_next_dino, rec_next_sig = self.sample(batch_size=img_cur.shape[0], cond=cond, mask=None, denoise=True, unnormalize=False)

        img_next = preds  # b t c h w
        img_next = self.normalize(img_next)  # [-1, 1]
        with torch.no_grad():
            lc_next = self.vae.encode5(img_next.flatten(0, 1))
            lc_next = lc_next.detach()
        N, C = lc_next.shape[-2:]
        # z2 = z2.reshape(B, T, C, H, W).flatten(1, 2)  # B, T*4, H, W
        lc_next = lc_next.reshape(B, -1, N, C)
        lc_next_dino, lc_next_sig = lc_next.chunk(2, dim=2)

        b, tl, n, _ = track.shape
        # if vis is None:
        #     vis = torch.ones((b, tl, n)).to(track.device)
        #
        # track = self._preprocess_track(track)
        # vid = self._preprocess_vid(vid)
        # vis = self._preprocess_vis(vis)

        track_loss = F.mse_loss(rec_next_dino, lc_next_dino) + F.mse_loss(rec_next_sig, lc_next_sig)
        # img_loss = ((rec_cur - img_cur) ** 2).mean()
        loss = track_loss# + img_loss
        out_img = torch.cat([img_next, img_next], dim=1)  # b 2t c h w
        ret_dict = {
            "loss": loss.item(),
            "track_loss": track_loss.item(),
            # "img_loss": img_loss.item(),
        }

        if return_outs:
            return loss.sum(), ret_dict, (rec_next, rec_cur)
        return loss.sum(), ret_dict, out_img

    def forward_feature(self, his,
                        preds,
                        track,
                        task_emb,
                        action,
                        lbd_track,
                        lbd_img,
                        p_img,
                        return_outs=False,
                        vis=None):
        B, T, C, H, W = his.shape
        his = self.normalize(his)  # [-1, 1]
        img_cur = his[:, -1]  # b c h w
        cond = [his, task_emb, action]
        feature = self.sample_dynamic(batch_size=img_cur.shape[0], cond=cond, mask=None, denoise=True,
                                      unnormalize=False)
        return feature

    @torch.no_grad()
    def sample_dynamic(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True, unnormalize=False,
                       use_action=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                batch_size = cond.shape[0]
            elif isinstance(cond, list):
                batch_size = cond[0].shape[0]
            else:
                raise NotImplementedError("")
        # down_ratio = self.vae.down_ratio
        his, task_emb, action = cond
        B, T1, C, H, W = his.shape
        with torch.no_grad():
            lc_his = self.vae.encode4(his.flatten(0, 1))
            lc_his = lc_his.detach().reshape(B, T1, lc_his.shape[-2], lc_his.shape[-1])  # b*t, n, c
            # lc_cur = self.vae.encode4(img_cur)#.sample()  # b n c
        z2 = lc_his[:, -1]  # b n c
        # lc_his = lc_his[:, :-1]  # b t n c
        cond = [lc_his, task_emb, action]
        # video = video.split(3, dim=1)
        # video = torch.stack(video, dim=1)  # B, T, 3, H, W
        # B = video.shape[0]
        # img_cur = video[:, -1]
        # z2 = self.vae.encode4(img_cur)
        # z2 = z2.detach()
        # N, C = z2.shape[-2:]
        # z2 = z2.reshape(B, T, C, H, W).flatten(1, 2)  # B, T*4, H, W
        z2 = z2.unsqueeze(1).repeat(1, self.transformer.tube_size, 1, 1)  # B, T, N, C
        sample_fn = self.sample_fn
        z = sample_fn((batch_size, self.transformer.tube_size, self.seq_len, channels),
                      up_scale=up_scale, unnormalize=False, cond=cond, x_in=z2, denoise=denoise, mask=mask,
                      use_action=use_action)
        return z

    @torch.no_grad()
    def sample(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True, unnormalize=False,
               use_action=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                batch_size = cond.shape[0]
            elif isinstance(cond, list):
                batch_size = cond[0].shape[0]
            else:
                raise NotImplementedError("")
        # down_ratio = self.vae.down_ratio
        his, task_emb, action = cond
        B, T1, C, H, W = his.shape
        with torch.no_grad():
            lc_his = self.vae.encode5(his.flatten(0, 1))
            lc_his = lc_his.detach().reshape(B, T1, lc_his.shape[-2], lc_his.shape[-1])  # b*t, n, c
            lc_his_dino, lc_his_sig = lc_his.chunk(2, dim=2)
            # lc_cur = self.vae.encode4(img_cur)#.sample()  # b n c
        z1_prior = lc_his_dino[:, -1]  # b n c
        z2_prior = lc_his_sig[:, -1]
        # lc_his = lc_his[:, :-1]  # b t n c
        cond = [lc_his_dino, lc_his_sig, task_emb, action]
        # video = video.split(3, dim=1)
        # video = torch.stack(video, dim=1)  # B, T, 3, H, W
        # B = video.shape[0]
        # img_cur = video[:, -1]
        # z2 = self.vae.encode4(img_cur)
        # z2 = z2.detach()
        # N, C = z2.shape[-2:]
        # z2 = z2.reshape(B, T, C, H, W).flatten(1, 2)  # B, T*4, H, W
        z1_prior = z1_prior.unsqueeze(1).repeat(1, self.transformer.tube_size, 1, 1)  # B, T, N, C
        z2_prior = z2_prior.unsqueeze(1).repeat(1, self.transformer.tube_size, 1, 1)  # B, T, N, C
        sample_fn = self.sample_fn
        z1, z2 = sample_fn((batch_size, self.transformer.tube_size, self.seq_len//2, channels),
                      up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise, mask=mask, use_action=use_action)
        z1, z2 = z1 / self.dino_scale, z2 / self.siglip_scale
        z1 = torch.cumsum(z1, dim=1)
        z2 = torch.cumsum(z2, dim=1)
        z1 = z1_prior + z1
        z1 = z1.detach()
        z2= z2_prior + z2
        z2 = z2.detach()
        # if self.scale_by_std:
        #     z = 1. / self.scale_factor * z.detach()
        # elif self.scale_by_softsign:
        #     z = z / (1 - z.abs())
        #     z = z.detach()
        # print(z.shape)
        # x_rec = []
        # for _ in range(z.shape[1]):
        #     x_rec.append(self.vae.decode(z[:, _].to(torch.float32)))
        # x_rec = torch.stack(x_rec, dim=1)  # B, T, 3, H, W
        # x_rec = x_rec.clamp(-1., 1.)
        # x_rec = self.vae.decode(z.to(torch.float32))  # B, 3, H, W
        # if unnormalize:
        #     x_rec = self.unnormalize(x_rec)
        # x_rec = torch.clamp(x_rec, min=0., max=1.)
        return z1, z2

    @torch.no_grad()
    def sample_from_latent(self, batch_size=16, up_scale=1, cond=None, mask=None, denoise=True, unnormalize=False,
               use_action=True):
        image_size, channels = self.image_size, self.channels
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                batch_size = cond.shape[0]
            elif isinstance(cond, list):
                batch_size = cond[0].shape[0]
            else:
                raise NotImplementedError("")
        # down_ratio = self.vae.down_ratio
        lc_his, task_emb, action = cond
        lc_his_dino, lc_his_sig = lc_his.chunk(2, dim=2)
        cond = lc_his_dino, lc_his_sig, task_emb, action
        B, T1, N, C = lc_his_dino.shape
        # with torch.no_grad():
        #     lc_his = self.vae.encode4(his.flatten(0, 1))
        #     lc_his = lc_his.detach().reshape(B, T1, lc_his.shape[-2], lc_his.shape[-1])  # b*t, n, c
            # lc_cur = self.vae.encode4(img_cur)#.sample()  # b n c
        z1_prior = lc_his_dino[:, -1]  # b n c
        z2_prior = lc_his_sig[:, -1]  # b n c
        # lc_his = lc_his[:, :-1]  # b t n c
        # cond = [lc_his, task_emb, action]
        # video = video.split(3, dim=1)
        # video = torch.stack(video, dim=1)  # B, T, 3, H, W
        # B = video.shape[0]
        # img_cur = video[:, -1]
        # z2 = self.vae.encode4(img_cur)
        # z2 = z2.detach()
        # N, C = z2.shape[-2:]
        z1_prior = z1_prior.unsqueeze(1).repeat(1, self.transformer.tube_size, 1, 1)  # B, T, N, C
        z2_prior = z2_prior.unsqueeze(1).repeat(1, self.transformer.tube_size, 1, 1)  # B, T, N, C
        sample_fn = self.sample_fn2
        # z1, z2 = sample_fn((batch_size, self.transformer.tube_size, self.seq_len, channels),
        #               up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise, mask=mask,
        #               use_action=use_action)

        z1, z2 = sample_fn((batch_size, self.transformer.tube_size, self.seq_len // 2, channels),
                           up_scale=up_scale, unnormalize=False, cond=cond, denoise=denoise, mask=mask,
                           use_action=use_action)
        z1, z2 = z1 / self.dino_scale, z2 / self.siglip_scale
        z1 = torch.cumsum(z1, dim=1)
        z2 = torch.cumsum(z2, dim=1)
        z1 = z1_prior + z1
        z1 = z1.detach()
        z2 = z2_prior + z2
        z2 = z2.detach()
        z = torch.cat([z1, z2], dim=2)
        return z


    @torch.no_grad()
    def sample_fn(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False, mask=None, use_action=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        rho = 1.
        sigma_max = 1.
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
        # t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
        #             self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        if sampling_timesteps > 1:
            t_steps = (sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
                    step - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        else:
            t_steps = torch.Tensor([1., 0.], device=device)
        alpha = 1
        x_next_dino = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        x_next_sig = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur_dino = x_next_dino
            x_cur_sig = x_next_sig
            if cond is not None:
                pred = self.transformer(x_cur_dino, x_cur_sig, cond, t_cur.repeat(batch), mask, use_action=use_action,
                                        is_training=False)
            else:
                pred = self.transformer(x_cur_dino, x_cur_sig, t_cur.repeat(batch), mask, use_action=use_action,
                                        is_training=False)
            C_dino, noise_dino, C_sig, nosie_sig = pred[:4]
            C_dino, noise_dino, C_sig, nosie_sig = C_dino.to(torch.float64), noise_dino.to(torch.float64), C_sig.to(torch.float64), nosie_sig.to(torch.float64)
            # x0 = x_cur - C * t_cur - noise * t_cur
            x0_dino = x_cur_dino - C_dino * t_cur - noise_dino * t_cur
            x0_sig = x_cur_sig - C_sig * t_cur - nosie_sig * t_cur
            # d_cur = (x_cur - x0) / t_cur
            # x_next = x_cur + (t_next - t_cur) * d_cur
            x_next_dino = x0_dino + t_next * C_dino + t_next * noise_dino
            x_next_sig = x0_sig + t_next * C_sig + t_next * nosie_sig
            # d_cur = C + noise
            # x_next = x_cur + (t_next - t_cur) * d_cur
            # Apply 2-order correction.
            # if i < sampling_timesteps - 1:
            #     if cond is not None:
            #         pred = self.model(x_next, t_next, cond)
            #     else:
            #         pred = self.model(x_next, t_next)
            #     C_, noise_ = pred[:2]
            #     C_, noise_ = C_.to(torch.float64), noise_.to(torch.float64)
            #     d_next = C_ + noise_
            #     x_next = x_cur + (t_next - t_cur) * 0.5 * (d_cur + d_next)
        x_dino, x_sig = x_next_dino.to(torch.float32), x_next_sig.to(torch.float32)
        return x_dino, x_sig

    @torch.no_grad()
    def sample_fn2(self, shape, up_scale=1, unnormalize=True, cond=None, denoise=False, mask=None, use_action=False):
        batch, device, sampling_timesteps = shape[0], self.eps.device, self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        rho = 1.
        sigma_max = 1.
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float64, device=device)
        # t_steps = (self.sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
        #             self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        if sampling_timesteps > 1:
            t_steps = (sigma_max ** (1 / rho) + step_indices / (sampling_timesteps - 1) * (
                    step - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        else:
            t_steps = torch.Tensor([1., 0.], device=device)
        alpha = 1
        x_next_dino = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        x_next_sig = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur_dino = x_next_dino
            x_cur_sig = x_next_sig
            if cond is not None:
                pred = self.transformer(x_cur_dino, x_cur_sig, cond, t_cur.repeat(batch), mask, use_action=use_action,
                                        is_training=False)
            else:
                pred = self.transformer(x_cur_dino, x_cur_sig, t_cur.repeat(batch), mask, use_action=use_action,
                                        is_training=False)
            C_dino, C_sig = pred[-2:]
            C_dino, C_sig = C_dino.to(torch.float64), C_sig.to(torch.float64)
            # x0 = x_cur - C * t_cur - noise * t_cur
            noise_dino = (x_cur_dino - (t_cur - 1) * C_dino) / t_cur
            noise_sig = (x_cur_sig - (t_cur - 1) * C_sig) / t_cur

            x0_dino = x_cur_dino - C_dino * t_cur - noise_dino * t_cur
            x0_sig = x_cur_sig - C_sig * t_cur - noise_sig * t_cur
            # d_cur = (x_cur - x0) / t_cur
            # x_next = x_cur + (t_next - t_cur) * d_cur
            x_next_dino = x0_dino + t_next * C_dino + t_next * noise_dino
            x_next_sig = x0_sig + t_next * C_sig + t_next * noise_sig
        x_dino, x_sig = x_next_dino.to(torch.float32), x_next_sig.to(torch.float32)
        return x_dino, x_sig

    def forward_vis(self, vid, track, task_emb, p_img):
        """
        track: (b, tl, n, 2)
        vid: (b, t, c, h, w)
        """
        b = vid.shape[0]
        assert b == 1, "only support batch size 1 for visualization"

        H, W = self.img_size
        _vid = vid.clone()
        track = self._preprocess_track(track)
        vid = self._preprocess_vid(vid)

        rec_track, rec_patches = self.forward(vid, track, task_emb, p_img)
        track_loss = F.mse_loss(rec_track, track)
        img_loss = F.mse_loss(rec_patches, self._patchify(vid))
        loss = track_loss + img_loss

        rec_image = self._unpatchify(rec_patches)

        # place them side by side
        combined_image = torch.cat([vid[:, -1], rec_image[:, -1]], dim=-1)  # only visualize the current frame
        combined_image = self.img_unnormalizer(combined_image) * 255
        combined_image = torch.clamp(combined_image, 0, 255)
        combined_image = rearrange(combined_image, '1 c h w -> h w c')

        track = track.clone()
        rec_track = rec_track.clone()

        rec_track_vid = tracks_to_video(rec_track, img_size=H)
        track_vid = tracks_to_video(track, img_size=H)

        combined_track_vid = torch.cat([track_vid, rec_track_vid], dim=-1)

        _vid = torch.cat([_vid, _vid], dim=-1)
        combined_track_vid = _vid * .25 + combined_track_vid * .75

        ret_dict = {
            "loss": loss.sum().item(),
            "track_loss": track_loss.sum().item(),
            "img_loss": img_loss.sum().item(),
            "combined_image": combined_image.cpu().numpy().astype(np.uint8),
            "combined_track_vid": combined_track_vid.cpu().numpy().astype(np.uint8),
        }

        return loss.sum(), ret_dict

    def _patchify(self, imgs):
        """
        imgs: (N, T, 3, H, W)
        x: (N, L, patch_size**2 * T * 3)
        """
        N, T, C, img_H, img_W = imgs.shape
        p = self.img_proj_encoder.patch_size[0]
        assert img_H % p == 0 and img_W % p == 0

        h = img_H // p
        w = img_W // p
        x = imgs.reshape(shape=(imgs.shape[0], T, C, h, p, w, p))
        x = rearrange(x, "n t c h p w q -> n h w p q t c")
        x = rearrange(x, "n h w p q t c -> n (h w) (p q t c)")
        return x

    def _unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * T * 3)
        imgs: (N, T, 3, H, W)
        """
        p = self.img_proj_encoder.patch_size[0]
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = rearrange(x, "n (h w) (p q t c) -> n h w p q t c", h=h, w=w, p=p, q=p, t=self.frame_stack, c=3)
        x = rearrange(x, "n h w p q t c -> n t c h p w q")
        imgs = rearrange(x, "n t c h p w q -> n t c (h p) (w q)")
        return imgs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
