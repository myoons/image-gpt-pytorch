import yaml
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import ImageGPT, ImageGPTConfig

from utils.utils import quantize, image_to_sequence
from utils.torch_utils import set_optimizer, set_scheduler


if __name__ == '__main__':
    with open('configs/xs.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = ImageGPTConfig(config)

    model = ImageGPT(config)
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)

    dataset = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=config.batch_size)

    clusters = torch.from_numpy(np.load('clusters/cifar10_clusters.npy'))

    for i, (batch, _) in enumerate(loader):
        quantized_batch = quantize(batch, clusters)  # (batch_size, width, height)
        sequence = image_to_sequence(quantized_batch)  # (batch_size, width * height)
        logits = model(sequence)  # (bach_size, sequence_length, vocab_size)

        logits = logits.view(-1, logits.size(-1))
        label = sequence.view(-1)
        loss = F.cross_entropy(logits, label)

        loss.backward()
        optimizer.step()
