import os
import math
import torch
import logging
import platform
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)


def set_optimizer(config, model):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay_params = set()
    no_decay_params = set()
    decay_modules = torch.nn.Linear
    no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for module_name, module in model.named_modules():
        for param_name, parameter in module.named_parameters():
            full_param_name = f'{module_name}.{param_name}' if module_name else param_name

            if param_name.endswith('bias'):
                no_decay_params.add(full_param_name)
            elif param_name.endswith('weight') and isinstance(module, decay_modules):
                decay_params.add(full_param_name)
            elif param_name.endswith('weight') and isinstance(module, no_decay_modules):
                no_decay_params.add(full_param_name)
    # special case for start_of_image
    no_decay_params.add('start_of_image')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay_params & no_decay_params
    union_params = decay_params | no_decay_params
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, \
        f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay_params))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay_params))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
    return optimizer


def set_scheduler(config, optimizer):
    scheduler = {
        "scheduler": LambdaLR(
            optimizer, learning_rate_schedule(config.warmup_steps, config.total_steps)
        ),
        "interval": "step",
    }
    return scheduler


def learning_rate_schedule(warmup_steps, total_steps):
    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn


def select_device(args, batch_size=None):
    s = f'Image-GPT-PyTorch 🚀 '
    cpu = args.device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {args.device} requested'

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(args.device.split(',') if args.device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += 'CPU\n'

    if args.local_rank in [-1, 0]:
        print(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')
