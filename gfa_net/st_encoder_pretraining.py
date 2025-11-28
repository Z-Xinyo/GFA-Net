import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from model.ctrgcn import Model  #from model.ctrgcn_tiny import Model
from einops import rearrange

class SAM(nn.Module):
    def __init__(self, in_channels, num_persons) -> None:
        super().__init__()
        self.num_persons = num_persons
        self.linear = nn.Linear(in_channels * 2, in_channels)

    def forward(self, input_tensor):
        N, V, C = input_tensor.shape
        input_tensor = input_tensor.view(N, self.num_persons, -1, C)
        out1 = input_tensor.mean(1)  # or amax(1)
        temp = torch.chunk(input_tensor, 2, 1)
        out2 = torch.abs(temp[0].squeeze(1) - temp[1].squeeze(1))  #差异绝对值
        out = torch.cat([out1, out2], 2)
        out = self.linear(out)

        return out

class STEncoder(nn.Module):
    """Two branch encoder"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head) -> None:
        super().__init__()
        self.d_model = hidden_size
        self.gcn = Model()

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
        body0 = xp[:, :2, :]
        body1 = xp[:, 20, :].unsqueeze(1)
        head = xp[:, 2:4, :]
        body = torch.cat([body0, body1, head], 1)

        arm_left0 = xp[:, 5:8, :]
        arm_left1 = xp[:, 21:23, :]
        arm_left = torch.cat([arm_left0, arm_left1], 1)
        arm_right0 = xp[:, 9:12, :]
        arm_right1 = xp[:, 23:25, :]
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

        xt = self.t_embedding(xt)  # temporal domain
        xs = self.sam(self.s_embedding(xs))  # spatial domain

        xt_l = torch.chunk(xt, 4, dim=1)
        xs_l = self.sptialsplit(xs)

        vt = self.t_encoder(xt)
        vs = self.s_encoder(xs)
        vsl = self.sl_encoder[0](xs_l[0]).amax(1, keepdims=True)  #主干
        vtl = self.tl_encoder[0](xt_l[0]).amax(1, keepdims=True)

        for i in range(3):
            vc_i = self.tl_encoder[i + 1](xt_l[i + 1]).amax(1, keepdims=True)
            vtl = torch.cat([vtl, vc_i], dim=1)

        for i in range(4):
            vp_i = self.sl_encoder[i + 1](xs_l[i + 1]).amax(1, keepdims=True)
            vsl = torch.cat([vsl, vp_i], dim=1)

        vt = vt.amax(dim=1)  # global max pooling
        vs = vs.amax(dim=1)   # global max pooling
        vtl = self.tl1_encoder(vtl).amax(dim=1)
        vsl = self.sl1_encoder(vsl).amax(dim=1)

        return vt, vs, vtl, vsl


class PretrainingEncoder(nn.Module):
    """multi_granularity network + projectors"""

    def __init__(self, t_input_size, s_input_size, hidden_size, num_head, num_class=60):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size

        self.st_encoder = STEncoder(t_input_size, s_input_size, hidden_size, num_head)

        self.t_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.s_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.sl_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.tl_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.sd_proj = nn.Sequential(
            nn.Linear(self.d_model*2, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.td_proj = nn.Sequential(
            nn.Linear(self.d_model*2, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, x):
        vt, vs, vtl, vsl = self.st_encoder(x)
        vtd = torch.cat([vt, vtl], dim=1)
        vsd = torch.cat([vs, vsl], dim=1)

        # projection
        zs = self.s_proj(vs)
        zt = self.t_proj(vt)
        ztl = self.tl_proj(vtl)
        zsl = self.sl_proj(vsl)

        ztd = self.td_proj(vtd)
        zsd = self.sd_proj(vsd)

        return zt, zs, ztl, zsl, ztd, zsd

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

        if knn_eval:  # return last layer features during  KNN evaluation (action retrieval)
            return v
        else:
            return self.fc(v)
