import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from model.ctrgcn_tiny import Model
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, dim_feat, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_feat // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim_feat * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim_feat, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_attention(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

class Transformer_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1., mlp_out_ratio=1.,
                 qkv_bias=True, drop=0.1, attn_drop=0.1, act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.attn = Attention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class SAM(nn.Module):
    def __init__(self, in_channels, num_persons) -> None:
        super().__init__()
        self.num_persons = num_persons
        self.linear = nn.Linear(in_channels * 2, in_channels)

    def forward(self, input_tensor):
        N, V, C = input_tensor.shape
        input_tensor = input_tensor.view(N, self.num_persons, -1, C)
        out1 = input_tensor.mean(1)  # or amax(1)
        temp = torch.chunk(input_tensor, 2, 1) #样本内双人分开
        out2 = torch.abs(temp[0].squeeze(1) - temp[1].squeeze(1)) #不同人之间的差异（绝对差）
        out = torch.cat([out1, out2], 2)
        out = self.linear(out)

        return out

                        # 特征 当前个人
def generate_group_feature(xp, current_individual):

        groups = {}
        for i, group in enumerate(current_individual):
            if group not in groups:
                groups[group] = []
            groups[group].append(xp[:, i, :])

        # 使用stack而不是cat，以确保分组后的张量形状正确
        xs = [torch.stack(groups[g], dim=1) for g in sorted(groups.keys())]

        return xs


class STEncoder(nn.Module):
    """Two branch encoder"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head) -> None:
        super().__init__()
        self.d_model = hidden_size
        self.gcn = Model()
        #self.spatial_group = [3, 3, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0, 1, 1, 2, 2]

        # temporal and spatial branch embedding layers
        self.t_embedding = nn.Sequential(
            nn.Linear(t_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.s_embedding = nn.Sequential(
            nn.Linear(s_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )

        # persons feature aggregation
        self.sam = SAM(hidden_size, 2)

        encoder_layer = TransformerEncoderLayer(hidden_size, num_head, hidden_size, batch_first=True)

        self.t_encoder = TransformerEncoder(encoder_layer, 1)

        self.tl_encoder = nn.ModuleList()
        for i in range(4):
            self.tl_encoder.append(TransformerEncoder(encoder_layer, 1))

        self.tl1_encoder = TransformerEncoder(encoder_layer, 1)

        self.s_encoder = TransformerEncoder(encoder_layer, 1)

        self.sl_encoder = nn.ModuleList()
        for i in range(5):
            self.sl_encoder.append(TransformerEncoder(encoder_layer, 1))

        self.sl1_encoder = TransformerEncoder(encoder_layer, 1)

    def sptialsplit(self, xp):
        # spatial split
        xs = []
        body0 = xp[:, :2, :] #0 1
        body1 = xp[:, 20, :].unsqueeze(1)
        head = xp[:, 2:4, :] #2 3
        body = torch.cat([body0, body1, head], 1)

        arm_left0 = xp[:, 5:8, :] #5 6 7
        arm_left1 = xp[:, 21:23, :] #21 22
        arm_left = torch.cat([arm_left0, arm_left1], 1)
        arm_right0 = xp[:, 9:12, :] #9 10 11
        arm_right1 = xp[:, 23:25, :] #23 24
        arm_right = torch.cat([arm_right0, arm_right1], 1)
        leg_left = xp[:, 12:16, :]
        leg_right = xp[:, 16:20, :]

        xs.append(body)  # body
        xs.append(arm_left)  # left arm
        xs.append(arm_right)  # right arm
        xs.append(leg_left)  # left leg
        xs.append(leg_right)  # right leg

        return xs

    def forward(self, x):
        # N, C, T, V, M
        xt, xs = self.gcn(x)
        #xt = rearrange(x, 'n c t v m->n t (c v m)')
        #xs = rearrange(x, 'n c t v m->n (v m) (c t)')

        xt = self.t_embedding(xt)  # temporal domain
        xs = self.sam(self.s_embedding(xs))  # spatial domain

        xt_l = torch.chunk(xt, 4, dim=1)
        xs_l = self.sptialsplit(xs)

        vt = self.t_encoder(xt)
        vs = self.s_encoder(xs)
        vsl = self.sl_encoder[0](xs_l[0]).amax(1, keepdims=True)
        vtl = self.tl_encoder[0](xt_l[0]).amax(1, keepdims=True)

        for i in range(3):
            vc_i = self.tl_encoder[i + 1](xt_l[i + 1]).amax(1, keepdims=True)
            vtl = torch.cat([vtl, vc_i], dim=1)

        for i in range(len(xs_l)-1):
            vp_i = self.sl_encoder[i + 1](xs_l[i + 1]).amax(1, keepdims=True)
            vsl = torch.cat([vsl, vp_i], dim=1)

        vt = vt.amax(dim=1)
        vs = vs.amax(dim=1)
        vtl = self.tl1_encoder(vtl).amax(dim=1)
        vsl = self.sl1_encoder(vsl).amax(dim=1)

        return vt, vs, vtl, vsl

class PretrainingEncoder_student(nn.Module):
    """multi_granularity network + projectors"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head):
        super(PretrainingEncoder_student, self).__init__()

        self.d_model = hidden_size

        self.st_encoder = STEncoder(t_input_size, s_input_size, hidden_size, num_head)

        self.t_proj = nn.Linear(self.d_model, self.d_model*4)

        self.s_proj = nn.Linear(self.d_model, self.d_model*4)

        self.sl_proj = nn.Linear(self.d_model, self.d_model*4)

        self.tl_proj = nn.Linear(self.d_model, self.d_model*4)

    def forward(self, x):
        vt, vs, vtl, vsl = self.st_encoder(x)

        # projection
        zs = self.s_proj(vs)
        zt = self.t_proj(vt)
        ztl = self.tl_proj(vtl)
        zsl = self.sl_proj(vsl)

        return zt, zs, ztl, zsl

class DownstreamEncoder(nn.Module):
    """multi_granularity network + classifier"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head, num_class=60):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.st_encoder = STEncoder(t_input_size, s_input_size, hidden_size, num_head)

        # linear classifier
        self.fc = nn.Linear(4 * self.d_model, num_class)

    def forward(self, x, knn_eval=False):

        vt, vs, vtl, vsl = self.st_encoder(x)

        v = torch.cat([vt, vs, vtl, vsl], dim=1)

        # return v

        if knn_eval:  # return last layer features during  KNN evaluation (action retrieval)
            return v
        else:
            return self.fc(v)
