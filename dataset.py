from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import numpy as np
from torch.utils import data

def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def getDataLoader(dataset_path, batch_size, num_workers=2):
    transform = T.Compose([
                T.Resize(size=(512, 512)),
                T.RandomCrop(256),
                T.ToTensor(),
            ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=InfiniteSamplerWrapper(train_dataset), num_workers=num_workers)

    return dataloader
