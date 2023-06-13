import torch
from dataset import DataSet
import argparse
import os

SPLIT = (0.6, 0.2, 0.2)
# SPLIT_ZERO_SHOT = (0.75, 0.25)

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default=None)
parser.add_argument('--dimensions', nargs="*", type=int, default=[3, 3, 3],
                    help='Number of features for every perceptual dimension')
parser.add_argument('--game_size', type=int, default=10,
                    help='Number of target/distractor objects')

args = parser.parse_args()


data_set = DataSet(properties_dim=args.dimensions,
                    game_size=args.game_size,
                    device='cpu')

if not os.path.exists('data/'):
    os.makedirs('data/')

if args.path is None:
    path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ').ds')
else:
    path = args.path

with open(path, "wb") as f:
    torch.save(data_set, f)

print("Data set is saved as: " + path)