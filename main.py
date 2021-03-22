import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.distributed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ImageGPT, ImageGPTConfig
from utils.utils import quantize, image_to_sequence
from utils.torch_utils import set_optimizer, set_scheduler, select_device
from utils.check import check_git_status, check_requirements


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


if __name__ == '__main__':
    # arguments, config
    args = parse_args()
    with open('configs/xs.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = ImageGPTConfig(config)

    # Set DDP variables
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    if args.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    # DDP mode
    args.total_batch_size = args.batch_size
    device = select_device(args, batch_size=args.batch_size)
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.total_batch_size // args.world_size

    # model, optimizer, scheduler
    model = ImageGPT(config)
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)

    # DDP
    model = model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank])

    # dataloader, clusters
    dataset = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size)
    clusters = torch.from_numpy(np.load('clusters/cifar10_clusters.npy'))

    for i, (batch, _) in tqdm(enumerate(loader), total=len(loader)):
        quantized_batch = quantize(batch, clusters)  # (batch_size, width, height)
        sequence = image_to_sequence(quantized_batch)  # (batch_size, width * height)
        sequence = sequence.to(args.local_rank)
        logits = model(sequence)  # (bach_size, sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        label = sequence.view(-1)
        loss = F.cross_entropy(logits, label)

        loss.backward()
        optimizer.step()

    print(loss)
