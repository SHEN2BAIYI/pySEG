from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoraQKV(nn.Module):
    """
        正常的 QKV 是使用一个线性层先将输入转化成 QKV 三个向量，然后再进行拆分。
    =>
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
    =>
        这里主要进行第一部分的 Lora 改写。
    """
    def __init__(
            self, qkv: nn.Module,
            linear_a_q: nn.Module, linear_a_v: nn.Module,
            linear_b_q: nn.Module, linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_a_v = linear_a_v
        self.linear_b_q = linear_b_q
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)

        q_add = self.linear_b_q(self.linear_a_q(x))
        v_add = self.linear_b_v(self.linear_a_v(x))

        qkv[:, :, :, :self.dim] += q_add
        qkv[:, :, :, -self.dim:] += v_add

        return qkv


class LoraSAM(nn.Module):
    def __init__(
            self, sam: Sam, r: int, lora_year=None
    ):
        super().__init__()
        assert r > 0, "r must be greater than 0"
        self.sam = sam
        self.r = r
        self.lora_year = lora_year if lora_year is not None else list(
            range(len(sam.image_encoder.blocks))
        )

        # 禁用模型的所有梯度
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam.mask_decoder.parameters():
            param.requires_grad = False

        for index, blk in enumerate(sam.image_encoder.blocks):
            if index not in self.lora_year:
                continue

            ori_qkv = blk.attn.qkv
            self.dim = ori_qkv.in_features

            # 低秩
            linear_a_q = nn.Linear(self.dim, r, bias=False)
            linear_b_q = nn.Linear(r, self.dim, bias=False)

            linear_a_v = nn.Linear(self.dim, r, bias=False)
            linear_b_v = nn.Linear(r, self.dim, bias=False)

            # 覆盖
            blk.attn.qkv = _LoraQKV(ori_qkv, linear_a_q, linear_a_v, linear_b_q, linear_b_v)

    def reset_parameters(self):
        for index, blk in enumerate(self.sam.image_encoder.blocks):
            if index not in self.lora_year:
                continue

            # a -> xavier-uniform
            nn.init.xavier_uniform_(blk.attn.qkv.linear_a_q.weight)
            nn.init.xavier_uniform_(blk.attn.qkv.linear_a_v.weight)

            # b -> zeros
            nn.init.zeros_(blk.attn.qkv.linear_b_q.weight)
            nn.init.zeros_(blk.attn.qkv.linear_b_v.weight)
