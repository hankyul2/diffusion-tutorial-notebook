1. run on single-gpu
   ```python
   python3 train.py -c 0 --use-wandb
   ```
2. run on multi-gpu
   ```python
   torchrun --nproc_per_node=2 --master_port=12345 train.py -c 0,1 -lr 2e-4
   ```