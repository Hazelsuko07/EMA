import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

#Import metcirs
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from math import ceil
import tqdm

#Import class for images
from PIL import Image


from collections import namedtuple


class ImageFolder(datasets.ImageFolder):
    """Modified torchvision.datasets.ImageFolder class with additional options:
        Preload images.
        Train / validation split.
        Support for multiple directories.

    Args:
        folders (list): Path list.
        loader (callable): A function to load a sample given its path.
        preload_size (tuple): Size of images after loading.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        load_transform (callable, optional): A function/transform that takes in a loading sample and returns a transformed version.
        val_transform (callable, optional): A function/transform that takes in a vlalidate sample and returns a transformed version.
        val_indices (iterable, optional): List of indices for validate.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples.
        imgs_tensor (Tensor): Tensor with preload images.
    """

    def __init__(self, folders, loader, preload_size, transform=None, load_transform=None, val_transform=None,
                 val_indices=[]):

        imgs = None
        imgs_tensor = None

        self.load_transform = load_transform
        self.val_transform = val_transform
        self.val_indices = val_indices

        for root in folders:
            super(ImageFolder, self).__init__(root, transform=transform, loader=loader)

            # bar = tqdm(range(len(self.imgs)), desc="Loading images")

            imgs_tensor_tmp = torch.zeros([len(self.imgs), *preload_size])
            imgs_tmp = self.imgs

            for index in range(len(self.imgs)):
                if self.load_transform:
                    imgs_tensor_tmp[index, :, :, :] = self.load_transform(loader(self.imgs[index][0]))
                else:
                    imgs_tensor_tmp[index, :, :, :] = loader(self.imgs[index][0])

                # bar.update()

            if imgs is not None:
                imgs += imgs_tmp
                imgs_tensor = torch.cat((imgs_tensor, imgs_tensor_tmp))
            else:
                imgs = imgs_tmp
                imgs_tensor = imgs_tensor_tmp

            # bar.close()

        self.imgs = imgs
        self.imgs_tensor = imgs_tensor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index.

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.val_transform and index in self.val_indices:
            return self.val_transform(self.imgs_tensor[index]), self.imgs[index][1]
        if self.transform:
            return self.transform(self.imgs_tensor[index]), self.imgs[index][1]
        return self.imgs_tensor[index], self.imgs[index][1]

    def __len__(self):
        return len(self.imgs)

train_dir = './data/chest_xray/train'
val_dir = './data/chest_xray/val'
test_dir = './data/chest_xray/test'


def init_CXR(mode = 'train'):
    norm_transforms = transforms.Compose([transforms.Normalize([0.4823, 0.4823, 0.4823],
                                                               [0.2361, 0.2361, 0.2361])])
    load_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()])


    if mode == 'train':
        dataset = ImageFolder([train_dir], loader=Image.open, preload_size=(3, 224, 224),
                              transform=norm_transforms,
                              load_transform=load_transforms)
    else:
        dataset = ImageFolder([test_dir], loader = Image.open, preload_size = (3, 224, 224),
                                   transform=norm_transforms,
                                   load_transform = load_transforms)

    return dataset


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from matplotlib import patches, patheffects

    trainset= init_CXR(mode = 'train')

    batch_size=4
    train_dl = DataLoader(trainset, batch_size=batch_size,)

    def show_img(im, figsize=None, ax=None):
        if not ax:
            fig,ax = plt.subplots(figsize=figsize)
        ax.imshow(im,cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return ax
    def draw_outline(o, lw):
      o.set_path_effects([patheffects.Stroke(
          linewidth=lw, foreground='black'), patheffects.Normal()])

    def draw_rect(ax, b):
        patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
        draw_outline(patch, 4)
    def draw_text(ax, xy, txt, sz=14):
        text = ax.text(*xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='bold')
        draw_outline(text, 1)

    image, klass = next(iter(train_dl))
    fig, axes = plt.subplots(1, 4, figsize=(12, 2))
    for i,ax in enumerate(axes.flat):
    #     ima=image[i].numpy().transpose((1, 2, 0))
        ima=image[i][0]
        b = klass[i]
        ax = show_img(ima.numpy(), ax=ax)
        draw_text(ax, (0,0), b)

    plt.tight_layout()
    plt.show()

