import torch

class CosineScheduleWithWarmup:
    def __init__(self, optimizer, total_steps, warmup_steps):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cosine_steps = total_steps - warmup_steps
        self.lin_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_steps, eta_min=0)
        
    def step(self, current_step):
        if current_step >= self.warmup_steps:
            self.cosine_scheduler.step()
        else:
            self.lin_scheduler.step()