import argparse
import csv
import os
import pickle

import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp
from torchvision import models

from arch import MLP
from data_utils import COVIDxDataModule, MNISTDataModule
from MIA.mia_threshold import mia

parser = argparse.ArgumentParser(description='Run an experiment.')
parser.add_argument('--dim', type=int, default=256, help='hidden dim of MLP')
parser.add_argument('--nw', type=int, default=1, help='number of workers')
parser.add_argument('--type', type=str, default='base',
                    help='base | forget | decay |invalid|matchy')

parser.add_argument('--dataset', type=str,
                    default='MNIST', help='MNIST or COVID')
parser.add_argument('--nclass', type=int, default=2, help='number of classes')
parser.add_argument('--batch_size', metavar='B', type=int, default=500,
                    help='batch size')
parser.add_argument('--audit', type=str, default='ks',
                    help='which auditing method')
parser.add_argument('--k', type=int, default=0,
                    help='percentage of modified samples')
parser.add_argument('--cal_data', type=str, default='MNIST',
                    help='name of calibration data')
parser.add_argument('--cal_size', type=int, default=10000,
                    help='size of calibration data')
parser.add_argument('--fold', type=int, default=0)

parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--qsize', type=int, default=2000)

parser.add_argument('--mixup', type=int, default=0)
parser.add_argument('--use_own', dest='use_own', action='store_true')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = 10


if args.dataset == 'MNIST':
    queryset = MNISTDataModule(batch_size=args.batch_size, mode='query',
                               k=args.k, calset=args.cal_data, use_own=args.use_own, fold=args.fold)
elif args.dataset == 'COVIDx':
    queryset = COVIDxDataModule(batch_size=args.batch_size, mode='query',
                                k=args.k, calset=args.cal_data, use_own=args.use_own, fold=args.fold)
dataloader_query_train = queryset.train_dataloader()

# Yangsibo: we also need to load the calibration set, and a test set for the calibration set (e.g. the MNIST test set)
if args.audit == 'EMA':
    if args.dataset == 'MNIST':
        dataset = MNISTDataModule(batch_size=args.batch_size,
                                  mode='cal', k=args.k, calset=args.cal_data, use_own=args.use_own, fold=args.fold)
    elif args.dataset == 'COVIDx':
        dataset = COVIDxDataModule(batch_size=args.batch_size,
                                   mode='cal', k=args.k, calset=args.cal_data, use_own=args.use_own, fold=args.fold)
    dataloader_cal_train = dataset.train_dataloader()
    dataloader_cal_test = dataset.test_dataloader()

# initialize
if args.dataset == 'MNIST':
    model_train = MLP.MLP(28, args.dim, 10).to(device)
    model_cal = MLP.MLP(28, args.dim, 10).to(device)
elif args.dataset == 'COVIDx':
    model_train = models.resnet18(pretrained=False, num_classes=2).to(device)
    model_cal = models.resnet18(pretrained=False, num_classes=2).to(device)

# load weights
ckpt_name_base = ''
ckpt_name_cal = 'caldata={}_k={}_size={}'.format(
    args.cal_data, args.k, args.cal_size)
if args.mixup:
    ckpt_name_base += 'mixup_'
    ckpt_name_cal += 'mixup_'
if args.use_own:
    # Use the fold-specific calibration model
    ckpt_name_cal += f'useown_fold{args.fold}_'

if args.dataset == 'MNIST':
    model_train.load_state_dict(torch.load(
        f'./saves_new/MNIST/base/{ckpt_name_base}training_epoch{args.epoch}.pkl', map_location=torch.device(device)))
    model_cal.load_state_dict(torch.load(
        f'./saves_new/MNIST/cal/{ckpt_name_cal}training_epoch{args.epoch}.pkl', map_location=torch.device(device)))
elif args.dataset == 'COVIDx':
    model_train.load_state_dict(torch.load(
        f'./saves_new/COVIDx/base/{ckpt_name_base}training_epoch{args.epoch}.pkl', map_location=torch.device(device)))
    model_cal.load_state_dict(torch.load(
        f'./saves_new/COVIDx/cal/{ckpt_name_cal}training_epoch{args.epoch}.pkl', map_location=torch.device(device)))

model_train.to(device)
model_cal.to(device)

model_train.eval()
model_cal.eval()

if args.audit == 'EMA':
    cal_train_output = []  # outputs of train samples in cal set with the cal model
    cal_train_output_y = []
    cal_test_output = []  # outputs of test samples in cal set with the cal model
    cal_test_output_y = []
    query_output = []  # outputs of query samples in query set with the trained model
    query_output_y = []

    with torch.no_grad():
        for images, labels in dataloader_query_train:  # first, get query_output
            images = images.to(device)
            labels = labels.to(device)
            if args.dataset == 'MNIST':
                out_train = torch.exp(model_train(images))
            else:
                out_train = torch.exp(F.log_softmax(
                    model_train(images), dim=-1))
            # print(out_train)
            query_output.append(out_train)
            query_output_y.append(labels)

        query_output_y = torch.cat(query_output_y).detach().cpu().numpy()

    query_output = torch.cat(query_output).detach().cpu().numpy()
    print(f'Finish query set, fold {args.fold}')
    with torch.no_grad():  # then, get thresholds for different metrics using the calibration set. we assume most of the cal set data are not included from the training model
        for images, labels in dataloader_cal_train:
            images = images.to(device)
            labels = labels.to(device)
            if args.dataset == 'MNIST':
                out_cal = torch.exp(model_train(images))
            else:
                out_cal = torch.exp(F.log_softmax(model_train(images), dim=-1))

            cal_train_output.append(out_cal)
            cal_train_output_y.append(labels)

        for images, labels in dataloader_cal_test:
            images = images.to(device)
            labels = labels.to(device)
            if args.dataset == 'MNIST':
                out_cal = torch.exp(model_train(images))
            else:
                out_cal = torch.exp(F.log_softmax(model_train(images), dim=-1))
            cal_test_output.append(out_cal)
            cal_test_output_y.append(labels)

        cal_train_output = torch.cat(cal_train_output).detach().cpu().numpy()
        cal_test_output = torch.cat(cal_test_output).detach().cpu().numpy()

        cal_train_output_y = torch.cat(
            cal_train_output_y).detach().cpu().numpy()
        cal_test_output_y = torch.cat(cal_test_output_y).detach().cpu().numpy()

        print(
            f'Size of query output {query_output.shape}, size of calibration train set {cal_train_output.shape}, size of calibration test set {cal_test_output.shape}')

    MIA = mia(cal_train_output, cal_train_output_y, cal_test_output,
              cal_test_output_y, query_output, query_output_y, num_classes=args.nclass)
    res = MIA._run_mia()
    print('Finish cal set')

    if not os.path.exists(f'./saves_new/EMA_{args.dataset}/'):
        os.makedirs(f'./saves_new/EMA_{args.dataset}/')
        os.makedirs(f'./saves_new/EMA_{args.dataset}/cal_set')
        os.makedirs(f'./saves_new/EMA_{args.dataset}/query_set')
        os.makedirs(f'./saves_new/EMA_{args.dataset}/thres')

    logname = f'caldata={args.cal_data}_epoch={args.epoch}_k={args.k}_calsize={args.cal_size}'
    if args.mixup:
        logname += '_mixup'
    if args.use_own:
        logname += '_useown'

    res['cal_values_bin'].to_csv(
        f'./saves_new/EMA_{args.dataset}/cal_set/binarized_{logname}.csv', index=False)
    res['cal_values'].to_csv(
        f'./saves_new/EMA_{args.dataset}/cal_set/continuous_{logname}.csv', index=False)

    res['query_values_bin'].to_csv(
        f'./saves_new/EMA_{args.dataset}/query_set/binarized_{logname}_fold{args.fold}.csv', index=False)
    res['query_values'].to_csv(
        f'./saves_new/EMA_{args.dataset}/query_set/continuous_{logname}_fold{args.fold}.csv', index=False)

    pickle.dump(res['thresholds'], open(
        f'./saves_new/EMA_{args.dataset}/thres/{logname}.pkl', 'wb'))
