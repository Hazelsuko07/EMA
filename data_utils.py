import os
import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader

from config import *
from datasets.covidxdataset import COVIDxDataset
from datasets.cxrdataset import init_CXR


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


transform_svhn = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnist_rotate = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnist_noise = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AddGaussianNoise(0., 3.)
])

transform_usps = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


DEFAULT_DATA_DIR = './data'
DEFAULT_NUM_WORKERS = 16


class MNISTDataModule():
    def __init__(self, batch_size: int = 32, data_dir: str = DEFAULT_DATA_DIR, num_workers: int = DEFAULT_NUM_WORKERS, k: float = 0, mode: str = 'base', calset: str = 'MNIST', use_own: bool = False, fold: int = 0):
        self.dir = data_dir
        self.data_dir = os.path.join(self.dir, 'mnist')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10

        self.k = k
        self.calset = calset
        self.use_own = use_own
        self.fold = fold
        self.mode = mode
        self.setup()

    def setup(self):
        # Assign train, test, cal, cal_test datasets for use in dataloaders
        if self.mode == 'base':
            trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                                  download=True, transform=transform_mnist)
            self.train_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['MNIST']['TRAINING_INDEX_START'], CONFIG['MNIST']['TRAINING_INDEX_END'])))
            self.test_set = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_mnist)

        elif self.mode == 'query' or (self.mode == 'cal' and self.use_own):
            trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                                  download=True, transform=transform_mnist)
            if self.fold > 0 and self.fold <= CONFIG['MNIST']['FOLD_NUM']:
                self.train_set = torch.utils.data.Subset(
                    trainset, list(range((self.fold-1)*2000, self.fold*2000)))
            elif self.fold == 0:        # SVHN
                trainset_svhn = torchvision.datasets.SVHN(
                    root='./data/svhn', split='train', download=True, transform=transform_svhn)
                self.train_set = torch.utils.data.Subset(
                    trainset_svhn, list(range(0, 2000)))
            else:                       # Unincluded fold in training
                self.train_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['MNIST']['TRAINING_OUT_INDEX_START'], CONFIG['MNIST']['TRAINING_OUT_INDEX_END'])))

            self.test_set = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform_mnist)

        elif self.mode == 'cal':
            if self.calset == 'MNIST':
                trainset_ori = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                          download=True, transform=transform_mnist)
                trainset_rotate = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                             download=True, transform=transform_mnist_rotate)
                trainset_noise = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                                            download=True, transform=transform_mnist_noise)

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_rotate = Subset(trainset_rotate, list(
                    range(int((100-self.k)/100*len(trainset_rotate)), int((100-self.k//2)/100*len(trainset_rotate)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k//2)/100*len(trainset_noise)), len(trainset_noise))))
                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_rotate, trainset_noise])

                self.test_set = torchvision.datasets.MNIST(
                    root=self.data_dir, train=True, download=True, transform=transform_mnist)
                self.test_set = Subset(self.test_set, list(
                    range(0, len(self.train_set))))     # part of the train set
            elif self.calset == 'FMNIST':
                trainset_ori = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                 download=True, transform=transform_mnist)
                trainset_rotate = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                    download=True, transform=transform_mnist_rotate)
                trainset_noise = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir, 'fmnist'), train=False,
                                                                   download=True, transform=transform_mnist_noise)

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_rotate = Subset(trainset_rotate, list(
                    range(int((100-self.k)/100*len(trainset_rotate)), int((100-self.k//2)/100*len(trainset_rotate)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k//2)/100*len(trainset_noise)), len(trainset_noise))))
                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_rotate, trainset_noise])

                self.test_set = torchvision.datasets.FashionMNIST(
                    root=self.data_dir, train=True, download=True, transform=transform_mnist)
                self.test_set = Subset(self.test_set, list(
                    range(0, len(self.train_set))))     # part of the train set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class COVIDxDataModule():
    def __init__(self, batch_size: int = 32, data_dir: str = DEFAULT_DATA_DIR, num_workers: int = DEFAULT_NUM_WORKERS, k: float = 0, mode: str = 'base', calset: str = 'COVID', use_own: bool = False, fold: int = 0):
        self.dir = data_dir
        self.data_dir = os.path.join(self.dir, 'covid')
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.k = k
        self.calset = calset
        self.use_own = use_own
        self.fold = fold
        self.mode = mode
        self.setup()

    def setup(self):
        if self.mode == 'base':
            trainset = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                     dim=(224, 224))
            self.train_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['COVID']['TRAINING_INDEX_START'], CONFIG['COVID']['TRAINING_INDEX_END'])))
            self.test_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['COVID']['TRAINING_TEST_INDEX_START'], CONFIG['COVID']['TRAINING_TEST_INDEX_END'])))
            print(len(self.train_set))

        elif self.mode == 'query' or (self.mode == 'cal' and self.use_own):
            trainset = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                     dim=(224, 224))
            if self.fold > 0 and self.fold <= CONFIG['COVID']['FOLD_NUM']:
                print((self.fold-1)*800, self.fold*800)
                self.train_set = torch.utils.data.Subset(
                    trainset, list(range((self.fold-1)*800, self.fold*800)))
            elif self.fold == 0:        # SVHN, TODO...
                trainset_RSNA = init_CXR(mode='test')
                rand_idx = np.random.randint(0, len(trainset_RSNA), 800)
                self.train_set = torch.utils.data.Subset(
                    trainset_RSNA, rand_idx)
            else:                       # Unincluded fold in training
                self.train_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['COVID']['TRAINING_OUT_INDEX_START'], CONFIG['COVID']['TRAINING_OUT_INDEX_END'])))

            self.test_set = torch.utils.data.Subset(trainset, list(range(
                CONFIG['COVID']['CAL_TEST_INDEX_START'], CONFIG['COVID']['CAL_TEST_INDEX_END'])))

        elif self.mode == 'cal':
            if self.calset == 'COVIDx':
                trainset = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                         dim=(224, 224))
                trainset_ori = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['COVID']['CAL_TRAIN_INDEX_START'], CONFIG['COVID']['CAL_TRAIN_INDEX_END'])))

                trainset_noise = COVIDxDataset(n_classes=2, dataset_path='./data/covid/',
                                               dim=(224, 224), noise=True)
                trainset_noise = torch.utils.data.Subset(trainset_noise, list(range(
                    CONFIG['COVID']['CAL_TRAIN_INDEX_START'], CONFIG['COVID']['CAL_TRAIN_INDEX_END'])))

                trainset_ori = Subset(trainset_ori, list(
                    range(0, int((100-self.k)/100*len(trainset_ori)))))
                trainset_noise = Subset(trainset_noise, list(
                    range(int((100-self.k)/100*len(trainset_noise)), len(trainset_noise))))

                self.train_set = ConcatDataset(
                    [trainset_ori, trainset_noise])

                self.test_set = torch.utils.data.Subset(trainset, list(range(
                    CONFIG['COVID']['CAL_TEST_INDEX_START'], CONFIG['COVID']['CAL_TEST_INDEX_END'])))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
