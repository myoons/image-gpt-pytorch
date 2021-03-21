import torchvision
import numpy as np
from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans


cifar10 = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=ToTensor())
cifar10_np = np.stack([x.numpy() for x, _ in cifar10])
cifar10_np = cifar10_np.transpose((0, 2, 3, 1))


def cluster_colors(images, num_clusters=6):
    pixels = images.reshape(-1, images.shape[-1])
    print(pixels.shape)
    print(pixels[0])
    input()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, verbose=1).fit(pixels)
    np.save('clusters/cifar10_clusters.npy', kmeans.cluster_centers_)


cluster_colors(cifar10_np)