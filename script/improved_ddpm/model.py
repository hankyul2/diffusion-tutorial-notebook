import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


def time_embed(timestep, dim, max_period=10000):
    """sinusoidal embedding for timestamp"""
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(start=0.0, end=half) / half).to(device=timestep.device)
    args = timestep[:, None].float() * freq
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def init_zero(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time, drop, norm=partial(nn.GroupNorm, 32), act=partial(nn.SiLU, inplace=False)):
        super().__init__()
        self.cv1 = nn.Sequential(norm(in_ch), act(), nn.Conv2d(in_ch, out_ch, 3, padding=1), norm(out_ch))
        self.cv2 = nn.Sequential(act(), nn.Dropout(drop), init_zero(nn.Conv2d(out_ch, out_ch, 3, padding=1)))
        self.time = nn.Sequential(act(), nn.Linear(time, out_ch * 2))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, time):
        out = self.cv1(x)
        scale, shift = self.time(time)[..., None, None].tensor_split(2, dim=1)
        out = self.cv2(out * (1 + scale) + shift) + self.skip(x)
        return out


class Attention(nn.Module):
    def __init__(self, in_ch, head, norm=partial(nn.GroupNorm, 32)):
        super().__init__()
        self.h = head
        self.div = (in_ch // head) ** -0.5

        self.norm = norm(in_ch)
        self.qkv = nn.Conv2d(in_ch, in_ch * 3, 1)
        self.proj = init_zero(nn.Conv2d(in_ch, in_ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).tensor_split(3, dim=1)
        q, k, v = [s.reshape(B, self.h, -1, H * W).permute(0, 1, 3, 2) for s in qkv]

        attn = F.softmax(q @ k.transpose(-1, -2) * self.div, dim=-1)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, -1, H, W)
        out = self.proj(out)

        return out + x


class ResAttn(nn.Module):
    def __init__(self, in_ch, out_ch, time, drop, head):
        super().__init__()
        self.conv = ResBlock(in_ch, out_ch, time, drop)
        self.attn = Attention(out_ch, head)

    def forward(self, x, time):
        out = self.conv(x, time)
        out = self.attn(out)

        return out


class Encoder(nn.Module):
    def __init__(self, head=4, layer=3, drop=0.3, time=512,
                 dim=(128, 256, 512, 1024), attn=(0, 1, 1, 1), stride=(0, 1, 1, 1)):
        super().__init__()
        self.layers = nn.ModuleList()
        for stage in range(len(dim)):
            prev_ch = dim[stage-1] if stage else dim[0]

            if stride[stage]:
                self.layers.append(nn.Conv2d(prev_ch, prev_ch, 3, 2, 1))

            if attn[stage]:
                fn = partial(ResAttn, head=head)
            else:
                fn = ResBlock

            self.layers.extend([fn(dim[stage] if _ else prev_ch, dim[stage], time, drop) for _ in range(layer)])

    def forward(self, x, time):
        result = [x]
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                result.append(layer(result[-1]))
            else:
                result.append(layer(result[-1], time))

        return result


class Decoder(nn.Module):
    def __init__(self, head=4, layer=3, drop=0.3, time=512,
                 dim=(1024, 512, 256, 128), attn=(0, 1, 1, 0), stride=(1, 1, 1, 0)):
        super().__init__()
        self.layers = nn.ModuleList()
        for stage in range(len(dim)):
            prev_ch = dim[stage-1] if stage else dim[0]
            next_ch = dim[stage+1] if stage < len(dim) - 1 else dim[-1]
            in_ch = prev_ch + dim[stage] if stage else dim[stage] * 2

            if attn[stage]:
                fn = partial(ResAttn, head=head)
            else:
                fn = ResBlock

            self.layers.extend([fn(dim[stage] * 2 if _ else in_ch, dim[stage], time, drop) for _ in range(layer)])
            self.layers.append(fn(dim[stage] + next_ch, dim[stage], time, drop))

            if stride[stage]:
                self.layers.append(nn.Conv2d(dim[stage], dim[stage], 3, 1, 1))

    def forward(self, x, enc, time):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                x = layer(x)
            else:
                x = torch.cat([x, enc.pop()], dim=1)
                x = layer(x, time)

        return x


class Middle(nn.Module):
    def __init__(self, hidden, time, drop, head):
        super().__init__()
        self.conv_attn = ResAttn(hidden, hidden, time, drop, head)
        self.conv = ResBlock(hidden, hidden, time, drop)

    def forward(self, x, time):
        out = self.conv_attn(x, time)
        out = self.conv(out, time)

        return out


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.hidden_dim = hidden_dim

    def forward(self, timestep):
        t = time_embed(timestep, self.hidden_dim)
        t = self.fc2(self.act(self.fc1(t)))
        return t


class AttnUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=6, layer=3, head=4, drop=0.3,
                 dim=(128, 256, 512, 1024), attn=(0, 1, 1, 0), stride=(0, 1, 1, 1),
                 norm=partial(nn.GroupNorm, 32), act=partial(nn.SiLU, inplace=True)):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dim[0], kernel_size=3, stride=1, padding=1)

        self.time_dim = dim[0]
        self.time_emb_dim = dim[-2]
        self.time_embed = TimeEmbedding(self.time_dim, self.time_emb_dim)

        self.enc = Encoder(head, layer, drop, self.time_emb_dim, dim, attn, stride)
        self.mid = Middle(dim[-1], self.time_emb_dim, drop, head)
        self.dec = Decoder(head, layer, drop, self.time_emb_dim, dim[::-1], attn[::-1], stride[::-1])

        self.fc = nn.Sequential(norm(dim[0]), act(), init_zero(nn.Conv2d(dim[0], out_ch, 3, 1, 1)))

    def forward(self, x, timestep):
        x = self.stem(x)
        t = self.time_embed(timestep)
        enc = self.enc(x, t)
        mid = self.mid(enc[-1], t)
        dec = self.dec(mid, enc, t)
        out = self.fc(dec)
        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    timestamp = torch.randint(2, (2,))
    f = AttnUNet()
    y = f(x, timestamp)
    print(y.shape)
