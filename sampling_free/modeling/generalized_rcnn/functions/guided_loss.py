import torch


class GuidedLoss(torch.nn.Module):
    def __init__(self, num_losses, warmup_iters=100):
        super().__init__()
        self.register_buffer("iters", torch.tensor(0.))
        self.register_buffer("ratio", torch.tensor(0.))
        self.sigmas = torch.nn.Parameter(torch.ones(num_losses))
        self.num_losses = num_losses
        self.warmup_iters = warmup_iters
    
    def forward(self, losses, names):
        if self.iters < self.warmup_iters:
            with torch.no_grad():
                self.ratio = sum(losses[1:]) / (self.num_losses - 1) / losses[0]
            self.iters += 1
        
        losses[0] = self.ratio * losses[0]

        weighted_losses = {
            name: loss/(2*sigma**2) for loss, sigma, name in zip(losses, self.sigmas, names)
        }
        
        weighted_losses.update({"|".join(names): self.sigmas.prod().log()})
        
        return weighted_losses
        
