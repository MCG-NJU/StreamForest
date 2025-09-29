# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import torch.nn as nn
from typing import Callable, Tuple
from transformers.integrations import is_deepspeed_zero3_enabled
from functools import partial

import numpy as np
import warnings

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from transformers.integrations import is_deepspeed_zero3_enabled

import time
import math
import random

from .memory_manager import MemoryManager, bipartite_soft_matching, merge_wavg

# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# MVD: https://github.com/ruiwang2021/mvd/blob/main/modeling_finetune.py
# --------------------------------------------------------
def get_spatial_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    
    pos_embed_zero_padding = np.zeros((pos_embed_spatial.shape[0],pos_embed_spatial.shape[1],embed_dim-embed_dim_spatial), dtype = pos_embed_spatial.dtype)

    pos_embed = np.concatenate([pos_embed_zero_padding, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([1, grid_size, grid_size, embed_dim])  # [T*H*W, D]

    return pos_embed




def get_temporal_pos_embed(embed_dim, t_size):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_temporal = embed_dim // 4
    
    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )
    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    
    pos_embed_zero_padding = np.zeros((pos_embed_temporal.shape[0],pos_embed_temporal.shape[1],embed_dim-embed_dim_temporal), dtype = pos_embed_temporal.dtype)
    
    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_zero_padding], axis=-1)
    pos_embed = pos_embed.reshape([t_size, 1, 1, embed_dim])  # [T*H*W, D]

    return pos_embed




# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class ToMe_FSTW_PEMF(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.mm_hidden_size = config.mm_hidden_size
        self.hw = vision_cfg.image_size // vision_cfg.patch_size
        self.num_attention_heads = vision_cfg.num_attention_heads
        self.mlp = nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size))
        self.max_pos_hw = self.hw
        self.max_pos_num_frames = config.mm_pos_num_frames
        self.time_pos_embedding_window = 512
        print("<<<mm_projector time_pos_embedding_window>>> : ", self.time_pos_embedding_window)
        self._set_spatial_pos_cache(max_grid_size=self.max_pos_hw)
        self._set_temporal_pos_cache(max_t_size=self.time_pos_embedding_window)

    def _set_spatial_pos_cache(self, max_grid_size, device='cpu'):
        if is_deepspeed_zero3_enabled():
            device='cuda'
        pos_embed = torch.from_numpy(get_spatial_pos_embed(self.mm_hidden_size, max_grid_size)).float().to(device)
        self.register_buffer("spatial_pos_embed", pos_embed, persistent=False)

    def _adjust_spatial_pos_cache(self, new_grid_size, device):
        adjust_pos = False
        if new_grid_size > self.max_pos_hw:
            self.max_pos_hw = new_grid_size
            adjust_pos = True
        if adjust_pos:
            raise NotImplementedError(f"{new_grid_size}")
            self._set_spatial_pos_cache(max_grid_size=self.max_pos_hw, device=device)

    def get_spatial_pos_embed(self, new_grid_size, device):
        # self._adjust_spatial_pos_cache(new_grid_size, device)
        return self.spatial_pos_embed[:, :new_grid_size, :new_grid_size].reshape((1, new_grid_size * new_grid_size, -1))

    def _set_temporal_pos_cache(self, max_t_size, device='cpu'):
        if is_deepspeed_zero3_enabled():
            device='cuda'
        pos_embed = torch.from_numpy(get_temporal_pos_embed(self.mm_hidden_size, t_size=max_t_size)).float().to(device)
        print("pos_emb shape",pos_embed.shape)
        self.register_buffer("temporal_pos_embed", pos_embed, persistent=False)

    def _adjust_temporal_pos_cache(self, new_t_size, device):
        adjust_pos = False
        if new_t_size > self.max_pos_num_frames:
            self.max_pos_num_frames = new_t_size
            adjust_pos = True
        if adjust_pos:
            raise NotImplementedError(f"{new_t_size}")
            # self._set_temporal_pos_cache(max_t_size=self.max_pos_num_frames, device=device)

    def get_temporal_pos_embed(self, new_t_size, device):
        # self._adjust_temporal_pos_cache(new_t_size, device)
        tpe = self.temporal_pos_embed  # [W,1,1,C]
        if new_t_size > self.time_pos_embedding_window:
            reps = new_t_size // self.time_pos_embedding_window + 1
            tpe = tpe.repeat(reps, 1, 1, 1)[:new_t_size]  # 4D slice
        else:
            tpe = tpe[:new_t_size]  # 4D slice
        return tpe.reshape((new_t_size, 1, -1))

    def merge_tokens(self, x, target_num_token):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        
        if tmp_p == target_num_token:  #not compress
            return x
        
        assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
                
        
        head = self.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2) # [b, p, c//head]
            merge, _ = bipartite_soft_matching(
                metric, 
                r
            )
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # x = x.reshape(-1, c)  # 300, 1024
        return x

    def forward(self, x, local_num_frames, is_image=False): # 单帧49
        # print("is image: ", is_image)
        # raise ValueError("You are pooler!!!")
        dtype = x.dtype
        device = x.device
        height = width = self.hw
        assert height * width == x.shape[1] // local_num_frames, x.shape
        assert local_num_frames<=1, "<<<tome memory projector error>>> This memory not support loacl_num_frame > 1 !!!"
        total_num_frames = x.shape[0]
        
        image_tokens = [729, 128]
        
        if is_image:
            spatial_pos = self.get_spatial_pos_embed(new_grid_size=height, device=x.device).to(x.dtype).repeat(x.shape[0], 1, 1)
            x = x + spatial_pos
            random_number = random.randint(0, 1)
            num_image_tokens = image_tokens[random_number]
            x = self.merge_tokens(x, target_num_token=num_image_tokens)
            x = self.mlp(x)
            return x
        
        spatial_pos = self.get_spatial_pos_embed(new_grid_size=height, device=x.device).to(x.dtype).repeat(x.shape[0], 1, 1)
        temporal_pos = self.get_temporal_pos_embed(new_t_size=x.shape[0], device=x.device).to(x.dtype).repeat(1, height*width, 1)
        x = x + spatial_pos + temporal_pos
        # print("<<<Mark>>> x.shape before memory: ", x.shape)
        memory_manager = MemoryManager(self.mm_hidden_size, self.num_attention_heads).to(x.device)

        for frame_idx in range(x.shape[0]):
            memory_manager.update(x[frame_idx])
        
        # memory_manager._update_long_memory()
        
        x = memory_manager.get_memory_tokens()
        
        # print("<<<Mark>>> x.shape after memory: ", x.shape)
        
        x = self.mlp(x)
        # print('I am pooler', x.shape)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "tome_fstw_pemf"}


if __name__ == "__main__":
    from easydict import EasyDict as edict
    config = edict({"mm_hidden_size": 1152, "hidden_size": 512, "mm_pos_num_frames":1})
    vision_cfg = edict({"patch_size": 14, "image_size":384, "num_attention_heads":16})
    connector = ToMe_FSTW_PEMF(config=config, vision_cfg=vision_cfg)
    x = torch.rand((1024, 729, 1152))
    print(connector(x, 1).shape)