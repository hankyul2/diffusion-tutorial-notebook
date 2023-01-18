import math
import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as TVTF

from script.improved_ddpm.model import AttnUNet


class UniformTimeSampler:
    def __init__(self, timestep):
        self.weight = np.ones(timestep)

    def sample(self, batch_size, device):
        w = self.weight
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class Diffusion(nn.Module):
    def __init__(self, beta_schedule_name, time_step):
        super().__init__()
        self.beta = self.get_named_beta_schedule(beta_schedule_name, time_step)
        self.beta = torch.from_numpy(self.beta)
        self.alpha = 1 - self.beta
        self.register_buffer('alphas', torch.cumprod(self.alpha, dim=0)[:, None, None, None])

    def get_named_beta_schedule(self, schedule_name, timestep, beta_max=0.999):
        if schedule_name == 'linear':
            scale = 1000 / timestep
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, timestep, dtype=np.float)
        elif schedule_name == 'cosine':
            cos = lambda t: math.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
            return np.array([min(1 - cos((i+1)/timestep) / cos(i/timestep), beta_max) for i in range(timestep)])
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")

    def forward(self, x, e, t):
        return self.alphas[t] ** 0.5 * x + (1 - self.alphas[t]) ** 0.5 * e

    def kl_loss(self, noise_hat, beta_coeff_hat, x, x_t, t):
        return 0


class Metric:
    def __init__(self, header='', fmt='{val:.4f} ({avg:.4f})'):
        self.val = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()
        self.val = val
        self.sum += val * n
        self.n += n
        self.avg = self.sum / self.n

    def compute(self):
        return self.avg

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)


def train(train_dataloader, model, uniform_time_sampler, diffusion, optimizer, epoch, args):
    simple_loss_m = Metric(header="Simple Loss:")
    kl_loss_m = Metric(header="KL Loss:")
    loss_m = Metric(header="Loss:")
    total_iter = len(train_dataloader)
    num_digits = len(str(total_iter))

    for batch_idx, (x, _) in enumerate(train_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        t, _ = uniform_time_sampler.sample(batch_size, args.device)
        noise = torch.randn_like(x).to(args.device)

        x_t = diffusion(x, noise, t).float()
        noise_hat, beta_coeff_hat = model(x_t, t).tensor_split(2, dim=1)

        simple_loss = ((noise_hat - noise) ** 2).mean()
        kl_loss = diffusion.kl_loss(noise_hat, beta_coeff_hat, x, x_t, t)
        loss = simple_loss + kl_loss * 0.001

        simple_loss_m.update(simple_loss, batch_size)
        kl_loss_m.update(kl_loss, batch_size)
        loss_m.update(loss, batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            print(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {simple_loss_m} {kl_loss_m} {loss_m}")


def run(args):
    train_transform = TVTF.Compose([TVTF.ToTensor(), TVTF.Normalize(args.mean, args.std)])
    train_dataset = CIFAR10('data', train=True, download=True, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = AttnUNet().to(args.device)
    uniform_time_sampler = UniformTimeSampler(args.diffusion_steps)
    diffusion = Diffusion(args.beta_schedule, args.diffusion_steps).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, 1+args.epoch):
        train(train_dataloader, model, uniform_time_sampler, diffusion, optimizer, epoch, args)


if __name__ == '__main__':
    class args:
        device = 'cuda:0'
        print_freq = 40

        diffusion_steps = 4000
        beta_schedule = 'cosine'

        epoch = 10
        lr = 1e-6
        weight_decay = 0

        mean = [0.5] * 3
        std = [0.5] * 3
        batch_size = 128
        num_workers = 4

    run(args)