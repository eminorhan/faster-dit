# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from collections import OrderedDict
from functools import partial
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True):

        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """Scaled dot-product attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, norm_layer=nn.LayerNorm):
        
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v)  # flash attention-2

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, use_conv=False):

        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
        ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=norm_layer)

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class TAE(nn.Module):
    """Transformer-based AE with ViT encoder-decoder"""
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            vocab_size=16,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm
        ):
       
        super().__init__()

        # --------------------------------------------------------------------------
        # QVAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.dict_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # QVAE decoder specifics
        self.decoder_embed = nn.Linear(vocab_size, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_embed
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # project to discrete codebook
        x = self.dict_proj(x)
        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean()  # mean loss per pixel
        return loss

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred)
        return loss, pred


class VITForRecognition(nn.Module):
    """PatchEmbedless VIT based on TAE"""
    def __init__(
            self,
            num_patches=256,
            vocab_size=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
            num_classes=None
        ):
       
        super().__init__()

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(vocab_size, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, num_classes, bias=True) if num_classes != None else nn.Identity()
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_embed
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def forward_head(self, x):
        x = x.mean(dim=1)  # global pooling 
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class VITForSegmentation(nn.Module):
    """PatchEmbedless VIT based on TAE"""
    def __init__(
            self,
            num_patches=256,
            patch_size=16,
            vocab_size=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
            num_classes=None
        ):
       
        super().__init__()

        self.aux_depth = int(decoder_depth * 0.75)
        self.patch_size = patch_size
        self.num_classes = num_classes

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(vocab_size, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.aux_decoder_norm = norm_layer(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, patch_size**2 * num_classes, bias=True)
        self.aux_head = nn.Linear(decoder_embed_dim, patch_size**2 * num_classes, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_embed
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.num_classes))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.num_classes, h * p, h * p))
        return imgs

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # result will hold both "out" and "aux"
        result = OrderedDict()

        # apply transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            if i+1 == self.aux_depth:
                aux = self.aux_head(self.aux_decoder_norm(x))  # auxiliary head
                aux = self.unpatchify(aux)

        x = self.decoder_norm(x)
        x = self.head(x)  # [N, L, P*P*C]
        x = self.unpatchify(x)  # [N, C, H, W]

        result["out"] = x
        result["aux"] = aux

        return result

# ============================== MODEL ARCHITECTURE LIST ==============================
# ************** TAE **************
# patch: 16 x 16
def tae_patch16_vocab16_px256():
    model = TAE(patch_size=16, vocab_size=16, img_size=256, embed_dim=1024, depth=15, num_heads=16, decoder_embed_dim=1024, decoder_depth=15, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch16_vocab64_px256():
    model = TAE(patch_size=16, vocab_size=64, img_size=256, embed_dim=1024, depth=15, num_heads=16, decoder_embed_dim=1024, decoder_depth=15, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch16_vocab256_px256():
    model = TAE(patch_size=16, vocab_size=256, img_size=256, embed_dim=1024, depth=15, num_heads=16, decoder_embed_dim=1024, decoder_depth=15, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

# patch: 32 x 32
def tae_patch32_vocab64_px256():
    model = TAE(patch_size=32, vocab_size=64, img_size=256, embed_dim=2048, depth=18, num_heads=32, decoder_embed_dim=2048, decoder_depth=18, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch32_vocab256_px256():
    model = TAE(patch_size=32, vocab_size=256, img_size=256, embed_dim=2048, depth=18, num_heads=32, decoder_embed_dim=2048, decoder_depth=18, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch32_vocab1024_px256():
    model = TAE(patch_size=32, vocab_size=1024, img_size=256, embed_dim=2048, depth=18, num_heads=32, decoder_embed_dim=2048, decoder_depth=18, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

# patch: 64 x 64
def tae_patch64_vocab256_px256():
    model = TAE(patch_size=64, vocab_size=256, img_size=256, embed_dim=2560, depth=21, num_heads=32, decoder_embed_dim=2560, decoder_depth=21, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch64_vocab1024_px256():
    model = TAE(patch_size=64, vocab_size=1024, img_size=256, embed_dim=2560, depth=21, num_heads=32, decoder_embed_dim=2560, decoder_depth=21, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch64_vocab4096_px256():
    model = TAE(patch_size=64, vocab_size=4096, img_size=256, embed_dim=2560, depth=21, num_heads=32, decoder_embed_dim=2560, decoder_depth=21, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

# patch: 128 x 128
def tae_patch128_vocab1024_px256():
    model = TAE(patch_size=128, vocab_size=1024, img_size=256, embed_dim=2560, depth=22, num_heads=32, decoder_embed_dim=2560, decoder_depth=22, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch128_vocab4096_px256():
    model = TAE(patch_size=128, vocab_size=4096, img_size=256, embed_dim=2560, depth=22, num_heads=32, decoder_embed_dim=2560, decoder_depth=22, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

def tae_patch128_vocab16384_px256():
    model = TAE(patch_size=128, vocab_size=16384, img_size=256, embed_dim=2560, depth=22, num_heads=32, decoder_embed_dim=2560, decoder_depth=22, decoder_num_heads=32, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model


# ************** RECOGNITION **************
# patch: 16 x 16
def vit_recognition_numpatches256_vocab16_base(num_classes=None):
    model = VITForRecognition(num_patches=256, vocab_size=16, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches256_vocab64_base(num_classes=None):
    model = VITForRecognition(num_patches=256, vocab_size=64, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches256_vocab256_base(num_classes=None):
    model = VITForRecognition(num_patches=256, vocab_size=256, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

# patch: 32 x 32
def vit_recognition_numpatches64_vocab64_base(num_classes=None):
    model = VITForRecognition(num_patches=64, vocab_size=64, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches64_vocab256_base(num_classes=None):
    model = VITForRecognition(num_patches=64, vocab_size=256, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches64_vocab1024_base(num_classes=None):
    model = VITForRecognition(num_patches=64, vocab_size=1024, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

# patch: 64 x 64
def vit_recognition_numpatches16_vocab256_base(num_classes=None):
    model = VITForRecognition(num_patches=16, vocab_size=256, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches16_vocab1024_base(num_classes=None):
    model = VITForRecognition(num_patches=16, vocab_size=1024, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches16_vocab4096_base(num_classes=None):
    model = VITForRecognition(num_patches=16, vocab_size=4096, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

# patch: 128 x 128
def vit_recognition_numpatches4_vocab1024_base(num_classes=None):
    model = VITForRecognition(num_patches=4, vocab_size=1024, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches4_vocab4096_base(num_classes=None):
    model = VITForRecognition(num_patches=4, vocab_size=4096, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_recognition_numpatches4_vocab16384_base(num_classes=None):
    model = VITForRecognition(num_patches=4, vocab_size=16384, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model


# ************** SEGMENTATION **************
# patch: 16 x 16
def vit_segmentation_numpatches256_vocab16_base(num_classes=None):
    model = VITForSegmentation(num_patches=256, vocab_size=16, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches256_vocab64_base(num_classes=None):
    model = VITForSegmentation(num_patches=256, vocab_size=64, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches256_vocab256_base(num_classes=None):
    model = VITForSegmentation(num_patches=256, vocab_size=256, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

# patch: 32 x 32
def vit_segmentation_numpatches64_vocab64_base(num_classes=None):
    model = VITForSegmentation(num_patches=64, vocab_size=64, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches64_vocab256_base(num_classes=None):
    model = VITForSegmentation(num_patches=64, vocab_size=256, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches64_vocab1024_base(num_classes=None):
    model = VITForSegmentation(num_patches=64, vocab_size=1024, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

# patch: 64 x 64
def vit_segmentation_numpatches16_vocab256_base(num_classes=None):
    model = VITForSegmentation(num_patches=16, vocab_size=256, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches16_vocab1024_base(num_classes=None):
    model = VITForSegmentation(num_patches=16, vocab_size=1024, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches16_vocab4096_base(num_classes=None):
    model = VITForSegmentation(num_patches=16, vocab_size=4096, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

# patch: 128 x 128
def vit_segmentation_numpatches4_vocab1024_base(num_classes=None):
    model = VITForSegmentation(num_patches=4, vocab_size=1024, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches4_vocab4096_base(num_classes=None):
    model = VITForSegmentation(num_patches=4, vocab_size=4096, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model

def vit_segmentation_numpatches4_vocab16384_base(num_classes=None):
    model = VITForSegmentation(num_patches=4, vocab_size=16384, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
    return model