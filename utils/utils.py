import torch


def squared_euclidean_distance(images, clusters):
    clusters = torch.transpose(clusters, 0, 1)
    squared_logits = torch.sum(torch.square(images), dim=1, keepdim=True)
    squared_clusters = torch.sum(torch.square(clusters), dim=0, keepdim=True)
    mul_logits_clusters = torch.matmul(images, clusters)
    euclidean_distance = squared_logits - 2 * mul_logits_clusters + squared_clusters
    return euclidean_distance


def quantize(images, clusters):
    batch_size, channels, height, width = images.shape
    images = images.permute(0, 2, 3, 1).contiguous()
    images = images.view(-1, channels)  # flatten to pixels
    distance = squared_euclidean_distance(images, clusters)
    quantized = torch.argmin(distance, 1)
    quantized = quantized.view(batch_size, height, width)
    return quantized


def unquantize(x, centroids):
    return centroids[x]


def image_to_sequence(images):
    pixels = images.view(images.shape[0], -1)  # flatten images into sequences
    return pixels


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']