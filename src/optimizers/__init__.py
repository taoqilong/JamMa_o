import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR
    if name == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_optimizer_tune(model, config):
    name = config.TRAINER.OPTIMIZER
    params = [
        {'params': model.detector.encoder.parameters(), 'lr': config.TRAINER.EN_LR},
        {'params': model.detector.decoder.parameters(), 'lr': config.TRAINER.DE_LR},
        {'params': model.matcher.parameters(), 'lr': config.TRAINER.DE_LR},
    ]
    if name == "adam":
        return torch.optim.Adam(params, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(params, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
