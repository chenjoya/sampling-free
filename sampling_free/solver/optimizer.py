import torch

def build_optimizer(cfg, model):
    params = []
    bias_lr = cfg.SOLVER.LR * cfg.SOLVER.BIAS_LR_FACTOR
    weight_decay_bias = cfg.SOLVER.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            params.append({"params": [value], "lr": bias_lr, "weight_decay": weight_decay_bias})
        else:
            params.append({"params": [value]})
    
    optimizer = torch.optim.SGD(
        params, 
        lr=cfg.SOLVER.LR, 
        momentum=cfg.SOLVER.MOMENTUM, 
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    return optimizer

