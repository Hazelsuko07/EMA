# import torch

import os
import numpy as np
import math
import sys
import urllib
import pickle
import input_data_class
import argparse
import csv
import time
sys.path.append('../')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from membership_inference_attacks import black_box_benchmarks

elapse_time = 0

def softmax_by_row(logits, T=1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator


def random_smooth(outputs, k, sigma, help_mix=None):
    start_time = time.time()
    avg_pred = np.zeros(outputs.shape)
    # Randomized smoothing
    # TODO: mix with training samples
    for _ in range(k):
        if help_mix is None:
            noise = np.stack(np.random.normal(0, sigma, o.shape)
                             for o in outputs)
        else:
            added = np.stack(o/np.linalg.norm(o, 2) for o in help_mix)  ### Normalization
            num_sample = len(outputs)
            noise = np.stack(np.random.normal(0, sigma, o.shape)
                             for o in outputs)
            noise = noise * added[np.random.randint(0, len(help_mix), num_sample)]
            # print(outputs)
        outputs_preturbed = outputs + noise
        avg_pred += outputs_preturbed
    # Done
    end_time = time.time()
    return avg_pred/k, (end_time - start_time)/len(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str,
                        default='location', help='location or texas')
    parser.add_argument('--predictions-dir', type=str,
                        default='./model', help='directory of saved predictions')
    parser.add_argument('--defended', type=int, default=1,
                        help='1 means defended; 0 means natural')
    parser.add_argument('--sigma', type=float, default=0,
                        help='std of the gaussian noise')
    parser.add_argument('--k', type=int, default=5,
                        help='number of repeated rounds of prediction')
    parser.add_argument('--smooth', type=int, default=1,
                        help='whether to apply rand smoothing')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--NN', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    input_data = input_data_class.InputData(dataset=dataset)
    (x_target, y_target, l_target) = input_data.input_data_attacker_evaluate()
    print(
        f'X shape: {x_target.shape}, y shape: {y_target.shape}, l shape: {l_target.shape}')
    npz_data = np.load(f'./model/{dataset}/saved_predictions/{dataset}_target_predictions.npz')
    if args.defended == 1:
        target_predictions = npz_data['defense_output']
    else:
        target_predictions = npz_data['tc_output']

    (x_shadow, y_shadow, l_shadow) = input_data.input_data_attacker_adv1()
    npz_data = np.load(f'./model/{dataset}/saved_predictions/{dataset}_shadow_predictions.npz')
    if args.defended == 1 and args.adaptive:
        shadow_predictions = npz_data['defense_output']
    else:
        shadow_predictions = npz_data['tc_output']

    print(shadow_predictions.shape)
    print(target_predictions.shape)

    if args.smooth and not args.defended == 1:
        if args.mixup:
            test_data = np.load(f'./model/{dataset}/saved_predictions/{dataset}_vanillatest_predictions.npz')['tc_output']
            help_data = test_data
            # help_data = np.concatenate((shadow_predictions, target_predictions, test_data))
            # print(help_data.shape)
            # print(shadow_predictions.shape)
            shadow_predictions, elapse_time = random_smooth(
                shadow_predictions, args.k, args.sigma, help_mix=help_data) ### Important: the model will take the same help data for both tests, it is hard-coded
            # shadow_predictions = np.zeros(shadow_predictions.shape)
            target_predictions, _ = random_smooth(
                target_predictions, args.k, args.sigma, help_mix=help_data)
        else:
            shadow_predictions, elapse_time = random_smooth(
                shadow_predictions, args.k, args.sigma, help_mix=None)
            # shadow_predictions = np.zeros(shadow_predictions.shape)
            target_predictions, _ = random_smooth(
                target_predictions, args.k, args.sigma, help_mix=None)
            # target_predictions = np.zeros(target_predictions.shape)

    shadow_train_performance = (
        shadow_predictions[l_shadow == 1], y_shadow[l_shadow == 1].astype('int32'))
    shadow_test_performance = (
        shadow_predictions[l_shadow == 0], y_shadow[l_shadow == 0].astype('int32'))
    target_train_performance = (
        target_predictions[l_target == 1], y_target[l_target == 1].astype('int32'))
    target_test_performance = (
        target_predictions[l_target == 0], y_target[l_target == 0].astype('int32'))

    print('Perform membership inference attacks!!!')
    if args.dataset == 'location':
        num_classes = 30
    else:
        num_classes = 100
    MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                               target_train_performance, target_test_performance, 
                               x_target, y_target,
                               x_shadow, y_shadow,
                               num_classes=num_classes)
    res = MIA._mem_inf_benchmarks(args=args)
    res.insert(0, args.sigma)
    res.insert(0, args.k)
    res.append(elapse_time*1000)

    # Save to log
    mode = args.dataset
    if args.mixup:
        logname = f'./result/{mode}_results_mix.csv'
    else:
        logname = f'./result/{mode}_results.csv'
    print(logname)
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter='\t')
            logwriter.writerow([
                'k', 'sigma', 'Train acc', 'Test acc', 'Correctness', 'Confidence', 'Entropy', 'Modified entropy', 'Best Attack Acc', 'Time per 1000 samples'
            ])

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter='\t')
        logwriter.writerow(res)
