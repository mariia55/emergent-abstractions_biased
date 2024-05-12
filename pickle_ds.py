import torch
from dataset import DataSet
import argparse
import os

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default=None)
##### add the granularity argument ######
parser.add_argument('--granularity', type=str, default='mixed',
                    help='Granularity of the context. Possible values: mixed, coarse and fine')
parser.add_argument('--dimensions', nargs="*", type=int, default=[3, 3, 3],
                    help='Number of features for every perceptual dimension')
parser.add_argument('--game_size', type=int, default=10,
                    help='Number of target/distractor objects')
parser.add_argument('--zero_shot', type=bool, default=False,
                    help='Set to True if zero-shot datasets should be generated.')
parser.add_argument("--save", type=bool, default=True)

args = parser.parse_args()

# prepare folder for saving
if args.path:
    if not os.path.exists(args.path + 'data/'):
        os.makedirs(args.path + 'data/')
else:
    if not os.path.exists('data/'):
        os.makedirs('data/')

# for normal dataset (not zero-shot)
### I added the granularity argument ###
if not args.zero_shot:
    data_set = DataSet(args.granularity,
                        args.dimensions,
                        game_size=args.game_size,
                        device='cuda')
    ### I added the granularity to file name to identify dataset ###
    if args.path:
        if data_set.granularity != 'mixed':
            path = (args.path + 'data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')_granularity_' + str(args.granularity) + '.ds')
        else:
            path = (args.path + 'data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ').ds')
    else:
        if data_set.granularity != 'mixed':
            path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')_granularity_' + str(args.granularity) + '.ds')
        else:
            path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ').ds')

    if args.save:
        with open(path, "wb") as f:
            torch.save(data_set, f)
        print("Data set is saved as: " + path)

# for zero-shot datasets
else:    
    for cond in ['generic', 'specific']:
        data_set = DataSet(args.granularity,
                           args.dimensions,
                           game_size=args.game_size,
                           testing=True, 
                           device='cuda')
        data_set = data_set.get_zero_shot_datasets(split_ratio=SPLIT_ZERO_SHOT, test_cond=cond)
        ### I added the granularity to file name do identify dataset ###
        if args.path:
            if data_set.granularity != 'mixed':
                path = (args.path + 'data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')_' + str(args.granularity) + '_' + str(cond) + '.ds')
            else:
                path = (args.path + 'data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')_' + str(cond) + '.ds')
        else:
            if data_set.granularity != 'mixed':
                path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')_' + str(args.granularity) + '_' + str(cond) + '.ds')
            else:
                path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ')_' + str(cond) + '.ds')
        if args.save:
            with open(path, "wb") as f:
                torch.save(data_set, f)
            print("Data set is saved as: " + path)
