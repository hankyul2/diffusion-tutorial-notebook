import os

import torch
from torch import distributed as dist
from torch.cuda.amp import GradScaler


class Metric:
    def __init__(self, header='', fmt='{val:.4f} ({avg:.4f})'):
        self.val = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt
        self.dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        if self.dist:
            self.world_size = int(os.environ['WORLD_SIZE'])

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()
        self.val = val
        self.sum += val * n
        self.n += n
        self.avg = self.sum / self.n

        if self.dist:
            self.avg = all_reduce_mean(self.sum, self.world_size) / self.n

    def compute(self):
        if self.dist:
            self.avg = all_reduce_mean(self.sum, self.world_size) / self.n

        return self.avg

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)


def all_reduce_mean(val, world_size):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    val = val / world_size
    return val


class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
