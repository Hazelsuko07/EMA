import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
train_transformer = transforms.Compose([
    # transforms.Resize(256),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


train_noise_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
    AddGaussianNoise(0., 3.)
])


def read_filepaths(file):
    paths, labels = [], []
    # print(file)
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if ('/ c o' in line):
                break

            subjid, path, label = line.split(' ')[:3]

            paths.append(path)
            labels.append(label)
    return paths, labels

class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, n_classes=3, dataset_path='./datasets', dim=(224, 224), noise=False):
        self.root = str(dataset_path) + 'COVID/'

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        file = os.path.join(dataset_path, 'allsingle.txt')
        self.paths, self.labels = read_filepaths(file)
        if noise:
            self.transform = train_noise_transformer
        else:
            self.transform = train_transformer
        # print("examples =  {}".format(len(self.paths)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(os.path.join(
            self.root, self.paths[index]), self.dim)
        label_tensor = torch.tensor(
            self.COVIDxDICT[self.labels[index]], dtype=torch.long)
        if self.CLASSES == 2:
            # reassign label of COVID to pneumonia.
            label_tensor[label_tensor == 2] = 0

        return image_tensor, label_tensor

    def load_image(self, img_path, dim):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        image_tensor = self.transform(image)

        return image_tensor


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import patches, patheffects
    from torch.utils.data import DataLoader

    trainset_COVID = COVIDxDataset(n_classes=2, dataset_path='../data/covid/',
                                   dim=(224, 224))
    batch_size = 100
    train_dl = DataLoader(trainset_COVID, batch_size=batch_size,)

    def show_img(im, figsize=None, ax=None):
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(im, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return ax

    def draw_outline(o, lw):
        o.set_path_effects([patheffects.Stroke(
            linewidth=lw, foreground='black'), patheffects.Normal()])

    def draw_rect(ax, b):
        patch = ax.add_patch(patches.Rectangle(
            b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
        draw_outline(patch, 4)

    def draw_text(ax, xy, txt, sz=14):
        text = ax.text(*xy, txt, verticalalignment='top',
                       color='white', fontsize=sz, weight='bold')
        draw_outline(text, 1)

    image, klass = next(iter(train_dl))
    fig, axes = plt.subplots(1, 4, figsize=(12, 2))
    for i, ax in enumerate(axes.flat):
        #     ima=image[i].numpy().transpose((1, 2, 0))
        ima = image[i][0]
        b = klass[i]
        ax = show_img(ima.numpy(), ax=ax)
        draw_text(ax, (0, 0), b)

    plt.tight_layout()
    plt.show()
