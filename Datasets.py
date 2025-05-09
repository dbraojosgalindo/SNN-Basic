import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BaseDataset:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = self.get_transform()

    def get_transform(self):
        raise NotImplementedError

    def get_loaders(self):
        raise NotImplementedError


class MNISTDataset(BaseDataset):
    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

    def get_loaders(self):
        train_set = datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        test_set = datasets.MNIST(self.data_path, train=False, download=True, transform=self.transform)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_loader, test_loader
