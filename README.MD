# Diffusion Tutorials


This repository contains diffusion tutorial notebooks with easy-to-follow-code implementations.
We hope this project help your understanding. Feel free to ask any questions.

## 1. Tutorial on diffusion via jupyter notebook



## 2. Tutorial on using pretrained model with simple code snippet

### 2.1. Sampling code snippets

```python
from ddpm import create_model

model = create_model('iddpm_cifar', pretrained=True)
model.sample(batch_size=16)
```

### 2.2. Pretrained model weight & log


### 2.3. Sampling results


## 3. Tutorial on train.py


`train.py` is more complicated and advanced than single jupyter notebook 
however, it is more flexible for running various experiment. 
So if you are considering to run various experiment, 
we recommend you to use `train.py`, which support followings features.


- Log on wandb
- Run on multiple-gpu
- Enable faster training using `amp` and `channels_last`.
- Enable trackable experiments by saving log on your computer.


1. install `ddpm`.

2. download `train.py`.

3. run on single-gpu
   ```python
   python3 train.py -c 0 
   ```
4. run on multi-gpu
   ```python
   torchrun --nproc_per_node=2 --master_port=12345 train.py -c 0,1 -lr 2e-4
   ```

## 4. References
