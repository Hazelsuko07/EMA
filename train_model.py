import argparse

import torch
import torch.nn as nn

from data_utils import COVIDxDataModule, MNISTDataModule
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--epoch', metavar='E', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', metavar='B', type=int, default=128,
                        help='batch size')
    parser.add_argument('--dataset', metavar='D', type=str, default='MNIST',
                        help='name of training data')

    parser.add_argument('--dim', type=int, default=256,
                        help='hidden dim of MLP')
    parser.add_argument('--add_layer', action='store_true')
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument('--k', type=int, default=0,
                        help='percentage of modified samples')

    parser.add_argument('--mode', type=str, default='base')
    parser.add_argument('--cal_data', type=str, default='MNIST',
                        help='name of calibration data')
    parser.add_argument('--mixup', type=int, default=0)
    parser.add_argument('--use_own', dest='use_own', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=2000)

    args = parser.parse_args()

    if args.mode == 'base':
        ckpt_name = ''
    elif args.mode == 'cal':
        ckpt_name = 'caldata={}_k={}_size={}'.format(
            args.cal_data, args.k, args.train_size)
    if args.mixup:
        ckpt_name += 'mixup_'
    if args.use_own:
        ckpt_name += f'useown_fold{args.fold}_'
    print(ckpt_name)

    if args.dataset == 'MNIST':
        MNIST_dataset = MNISTDataModule(batch_size=args.batch_size,
                                        mode=args.mode, k=args.k, calset=args.cal_data, use_own=args.use_own, fold=args.fold)
        MNIST_dataset.train_set = torch.utils.data.Subset(
            MNIST_dataset.train_set, list(range(0, args.train_size)))

        MNIST_trainer = Trainer(dataset=MNIST_dataset, name=args.dataset, dim=args.dim, criterion=nn.NLLLoss(
        ), max_epoch=args.epoch, mode=args.mode, ckpt_name=ckpt_name, mixup=args.mixup)

        MNIST_trainer.run()
    elif args.dataset == 'COVIDx':
        COVID_dataset = COVIDxDataModule(batch_size=args.batch_size,
                                         mode=args.mode, k=args.k, calset=args.cal_data, use_own=args.use_own, fold=args.fold)
        # COVID_dataset.train_set = torch.utils.data.Subset(
        #     COVID_dataset.train_set, list(range(0, args.train_size)))
        print(len(COVID_dataset.train_set))

        COVID_trainer = Trainer(dataset=COVID_dataset, name=args.dataset, dim=args.dim, criterion=nn.CrossEntropyLoss(
            reduction='mean'), max_epoch=args.epoch, mode=args.mode, ckpt_name=ckpt_name, mixup=args.mixup)

        COVID_trainer.run()
