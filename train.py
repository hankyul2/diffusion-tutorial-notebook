import math
import os
from argparse import ArgumentParser

import numpy as np

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms as TVTF

from ddpm.model_factory import create_model
from ddpm.setup import setup
from ddpm.utils import Metric, NativeScalerWithGradAccum


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
        return torch.tensor(1.0).to(noise_hat.device)


def train(train_dataloader, model, uniform_time_sampler, diffusion, optimizer, epoch, args, scaler=None, scheduler=None):
    simple_loss_m = Metric(header="Simple Loss:")
    kl_loss_m = Metric(header="KL Loss:")
    loss_m = Metric(header="Loss:")
    total_iter = len(train_dataloader)
    num_digits = len(str(total_iter))

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    for batch_idx, (x, _) in enumerate(train_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        t, _ = uniform_time_sampler.sample(batch_size, args.device)
        noise = torch.randn_like(x).to(args.device)

        x_t = diffusion(x, noise, t).float()
        if args.channels_last:
            x_t = x_t.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(args.amp):
            noise_hat, beta_coeff_hat = model(x_t, t).tensor_split(2, dim=1)

        simple_loss = ((noise_hat - noise) ** 2).mean()
        kl_loss = diffusion.kl_loss(noise_hat, beta_coeff_hat, x, x_t, t)
        loss = simple_loss + kl_loss * 0.001

        simple_loss_m.update(simple_loss, batch_size)
        kl_loss_m.update(kl_loss, batch_size)
        loss_m.update(loss, batch_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {simple_loss_m} {kl_loss_m} {loss_m}")

    loss_dict = {'train_simple': simple_loss_m.compute(),
                 "train_kl": kl_loss_m.compute(),
                 'train_total': loss_m.compute()}

    return loss_dict


@torch.no_grad()
def valid(val_dataloader, model, uniform_time_sampler, diffusion, epoch, args):
    simple_loss_m = Metric(header="Simple Loss:")
    kl_loss_m = Metric(header="KL Loss:")
    loss_m = Metric(header="Loss:")
    total_iter = len(val_dataloader)
    num_digits = len(str(total_iter))

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    for batch_idx, (x, _) in enumerate(val_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        t, _ = uniform_time_sampler.sample(batch_size, args.device)
        noise = torch.randn_like(x).to(args.device)

        x_t = diffusion(x, noise, t).float()
        if args.channels_last:
            x_t = x_t.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(args.amp):
            noise_hat, beta_coeff_hat = model(x_t, t).tensor_split(2, dim=1)

        simple_loss = ((noise_hat - noise) ** 2).mean()
        kl_loss = diffusion.kl_loss(noise_hat, beta_coeff_hat, x, x_t, t)
        loss = simple_loss + kl_loss * 0.001

        simple_loss_m.update(simple_loss, batch_size)
        kl_loss_m.update(kl_loss, batch_size)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            args.log(f"VALID({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {simple_loss_m} {kl_loss_m} {loss_m}")

    loss_dict = {'val_simple': simple_loss_m.compute(),
                 "val_kl": kl_loss_m.compute(),
                 'val_total': loss_m.compute()}

    return loss_dict


def run(args):
    setup(args)
    transform = TVTF.Compose([TVTF.ToTensor(), TVTF.Normalize(args.mean, args.std)])
    train_dataset = CIFAR10('data', train=True, download=True, transform=transform)
    val_dataset = CIFAR10('data', train=False, download=True, transform=transform)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=val_sampler)

    model = create_model(args.model_name).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    uniform_time_sampler = UniformTimeSampler(args.diffusion_steps)
    diffusion = Diffusion(args.beta_schedule, args.diffusion_steps).to(args.device)

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    best_loss = 1000.0
    for epoch in range(1, 1+args.epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        metric = {}
        metric.update(train(train_dataloader, ddp_model if ddp_model else model,
                           uniform_time_sampler, diffusion, optimizer, epoch, args, scaler))
        metric.update(valid(val_dataloader, model, uniform_time_sampler, diffusion, epoch, args))

        args.log(f"EPOCH({epoch:03}): " + " ".join([f"{k.upper()}: {v:.4f}" for k, v in metric.items()]))
        if args.use_wandb:
            args.log(metric, metric=True)

        if best_loss > metric['val_total']:
            best_loss = metric['val_total']
            if args.is_rank_zero:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{args.model_name}.pth'))
                args.log(f"Saved model (val loss: {best_loss:0.4f}) in to {args.log_dir}")


if __name__ == '__main__':
    def get_parser():
        parser = ArgumentParser()
        parser.add_argument('-m', '--model-name', default='iddpm', type=str)
        parser.add_argument('--dropout', default=0.3, type=float)
        parser.add_argument('--diffusion-steps', default=4000, type=int)
        parser.add_argument('--beta-schedule', default='cosine', type=str)

        parser.add_argument('-e', '--epoch', default=100, type=int)
        parser.add_argument('-lr', '--lr', default=1e-4, type=float)
        parser.add_argument('-wd', '--weight-decay', default=0, type=float)
        parser.add_argument('-b', '--batch-size', default=128, type=int)

        parser.add_argument('--mean', default=[0.5] * 3, type=float, nargs='+')
        parser.add_argument('--std', default=[0.5] * 3, type=float, nargs='+')
        parser.add_argument('--num-workers', default=4, type=int)

        parser.add_argument('--grad-norm', default=None, type=int)
        parser.add_argument('--grad-accum', default=1, type=int)
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--use-wandb', action='store_true')
        parser.add_argument('--amp', action='store_true')
        parser.add_argument('--channels-last', action='store_true')
        parser.add_argument('--output-dir', default='log', type=str)
        parser.add_argument('--project-name', default='learn-diffusion-model', type=str)
        parser.add_argument('--exp-name', default=None, type=str)
        parser.add_argument('--who', default='hankyul2', type=str)
        parser.add_argument('-c', '--cuda', default='0,', type=str)
        parser.add_argument('-p', '--print-freq', default=40, type=int)
        parser.set_defaults(amp=True, channels_last=True)

        return parser

    parser = get_parser()
    args = parser.parse_args()
    exp_options = ['model_name', 'diffusion_steps', 'beta_schedule', 'epoch', 'lr', 'batch_size']
    args.exp_name = "_".join([f"{k}_{getattr(args, k)}" for k in exp_options])
    run(args)