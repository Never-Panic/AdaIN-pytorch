from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T

def getDataLoader(dataset_path, batch_size, num_workers=2):
    transform = T.Compose([
                T.Resize(size=(512, 512)),
                T.RandomCrop(256),
                T.ToTensor(),
            ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
