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