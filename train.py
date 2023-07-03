# code based on https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/train.py
# and https://github.com/jayelm/emergent-generalization/blob/master/code/train.py

import argparse
import torch
#print(torch.__version__)
#import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import egg.core as core
from egg.core.language_analysis import TopographicSimilarity
# copy language_analysis_local from hierarchical_reference_game
from language_analysis_local import *
import os
import pickle

import dataset
from archs import Sender, Receiver
import itertools


SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


def get_params(params):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_dataset', type=str, default=None,
                        help='If provided that data set is loaded. Datasets can be generated with pickle.ds'
                            'This makes sense if running several runs with the exact same dataset.')
    parser.add_argument('--dimensions', nargs='+', type=int, default= [3, 3, 3])
    parser.add_argument('--attributes', type=int, default=3)
    parser.add_argument('--values', type=int, default=4)
    parser.add_argument('--game_size', type=int, default=10)
    parser.add_argument('--vocab_size_factor', type=int, default=3,
                        help='Factor applied to minimum vocab size to calculate actual vocab size')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Size of the hidden layer of Sender and Receiver,\
                             the embedding will be half the size of hidden ')
    parser.add_argument('--sender_cell', type=str, default='gru',
                        help='Type of the cell used for Sender {rnn, gru, lstm}')
    parser.add_argument('--receiver_cell', type=str, default='gru',
                        help='Type of the cell used for Receiver {rnn, gru, lstm}')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="Learning rate for Sender's and Receiver's parameters ")
    parser.add_argument('--temperature', type=float, default=1.5,
                        help="Starting GS temperature for the sender")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="linear cost term per message length")
    parser.add_argument('--temp_update', type=float, default=0.99,
                        help="Minimum is 0.5")
    parser.add_argument('--save', type=bool, default=False, help="If set results are saved")
    parser.add_argument('--num_of_runs', type=int, default=1, help="How often this simulation should be repeated")
    parser.add_argument('--zero_shot', type=bool, default=False,
                        help="If set then zero_shot dataset will be trained and tested")
    parser.add_argument('--zero_shot_test', type=str, default=None,
                        help="Must be either 'generic' or 'specific'.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Specifies the device for tensor computations. Defaults to 'cuda'.")
    parser.add_argument('--path', type=str, default="",
                        help="Path where to save the results - needed for running on HPC3.")
    parser.add_argument('--include_concept', type=bool, default=False,
                        help="Not implemented yet: If set to True, then full concepts will be created and preserved during training (opposed to preserving only targets and regenerating concepts after training)")
    parser.add_argument('--context_unaware', type=bool, default=False,
                        help="If set to True, then the speakers will be trained context-unaware, i.e. without access to the distractors.")

    args = core.init(parser, params)

    return args


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Loss needs to be defined for gumbel softmax relaxation.
    For a discriminative game, accuracy is computed by comparing the index with highest score in Receiver
    output (a distribution of unnormalized probabilities over target positions) and the corresponding 
    label read from input, indicating the ground-truth position of the target.
    Adaptation to concept game with multiple targets after Mu & Goodman (2021) with BCEWithLogitsLoss
        receiver_output: Tensor of shape [batch_size, n_objects]
        labels: Tensor of shape [batch_size, n_objects]
    """
    # after Mu & Goodman (2021):
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(receiver_output, labels)
    receiver_pred = (receiver_output > 0).float()
    per_game_acc = (receiver_pred == labels).float().mean(1).cpu().numpy()
    acc = per_game_acc.mean()
    return loss, {'acc': acc}


def train(opts, datasets, verbose_callbacks=False): 
    """
    Train function completely copied from hierarchical_reference_game.
    """

    if opts.save:
        # make folder for new run
        latest_run = len(os.listdir(opts.save_path))
        opts.save_path = os.path.join(opts.save_path, str(latest_run))
        os.makedirs(opts.save_path)
        pickle.dump(opts, open(opts.save_path + '/params.pkl', 'wb'))
        save_epoch = opts.n_epochs
    else:
        save_epoch = None

    train, val, test = datasets
    #print("train", train)
    dimensions = train.dimensions

    train = torch.utils.data.DataLoader(train, batch_size=opts.batch_size, shuffle=True)
    val = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(test, batch_size=opts.batch_size, shuffle=False)

    # initialize sender and receiver agents
    sender = Sender(opts.hidden_size, sum(dimensions), opts.game_size, opts.context_unaware)
    receiver = Receiver(sum(dimensions), opts.hidden_size)

    minimum_vocab_size = dimensions[0] + 1  # plus one for 'any'
    vocab_size = minimum_vocab_size * opts.vocab_size_factor + 1  # multiply by factor plus add one for eos-symbol

    # initialize game
    sender = core.RnnSenderGS(sender,
                              vocab_size,
                              int(opts.hidden_size / 2),
                              opts.hidden_size,
                              cell=opts.sender_cell,
                              max_len=len(dimensions),
                              temperature=opts.temperature)

    receiver = core.RnnReceiverGS(receiver,
                                  vocab_size,
                                  int(opts.hidden_size / 2),
                                  opts.hidden_size,
                                  cell=opts.receiver_cell)

    game = core.SenderReceiverRnnGS(sender, receiver, loss, length_cost=opts.length_cost)

    # set learning rates
    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.learning_rate},
        {'params': game.receiver.parameters(), 'lr': opts.learning_rate}
    ])

    # setup training and callbacks
    # results/ data set name/ kind_of_dataset/ run/
    callbacks = [SavingConsoleLogger(print_train_loss=True, as_json=True,
                                     save_path=opts.save_path, save_epoch=save_epoch),
                 core.TemperatureUpdater(agent=sender, decay=opts.temp_update, minimum=0.5)]
    if opts.save:
        callbacks.extend([core.callbacks.InteractionSaver([opts.n_epochs],
                                                          test_epochs=[opts.n_epochs],
                                                          checkpoint_dir=opts.save_path),
                          core.callbacks.CheckpointSaver(opts.save_path, checkpoint_freq=0)])
    if verbose_callbacks:
        callbacks.extend([
            TopographicSimilarityConceptLevel(dimensions, is_gumbel=True,
                                              save_path=opts.save_path, save_epoch=save_epoch),
            #MessageLengthHierarchical(len(dimensions),
            #                          print_train=True, print_test=True, is_gumbel=True,
            #                          save_path=opts.save_path, save_epoch=save_epoch)
        ])

    trainer = core.Trainer(game=game, optimizer=optimizer,
                           train_data=train, validation_data=val, callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs)

    # after training evaluate performance on the test data set
    if len(test):
        trainer.validation_data = test
        eval_loss, interaction = trainer.eval()
        acc = torch.mean(interaction.aux['acc']).item()
        print("test accuracy: " + str(acc))
        if opts.save:
            loss_and_metrics = pickle.load(open(opts.save_path + '/loss_and_metrics.pkl', 'rb'))
            loss_and_metrics['final_test_loss'] = eval_loss
            loss_and_metrics['final_test_acc'] = acc
            pickle.dump(loss_and_metrics, open(opts.save_path + '/loss_and_metrics.pkl', 'wb'))


def main(params):
    """
    Dealing with parameters and loading dataset. Copied from hierarchical_reference_game and adapted.
    """
    opts = get_params(params)

    # NOTE: I checked and the default device seems to be cuda
    # Otherwise there is an option in a later pytorch version (don't know about compatibility with egg):
    #torch.set_default_device(opts.device)

    # has to be executed in Project directory for consistency
    #assert os.path.split(os.getcwd())[-1] == 'emergent-abstractions'

    # dimensions calculated from attribute-value pairs:
    if not opts.dimensions:
        opts.dimensions = list(itertools.repeat(opts.values, opts.attributes))

    # if not opts.dimensions == [16, 16, 16, 16, 16]: # NOTE: just for hyperparameter search

    data_set_name = '(' + str(len(opts.dimensions)) + ',' + str(opts.dimensions[0]) + ')'
    folder_name = (data_set_name + '_game_size_' + str(opts.game_size) 
                        + '_vsf_' + str(opts.vocab_size_factor))
    folder_name = os.path.join("results", folder_name)

    # if name of precreated data set is given, load dataset
    if opts.load_dataset:
        data_set = torch.load(opts.path + 'data/' + opts.load_dataset)
        print('data loaded from: ' + 'data/' + opts.load_dataset)

    for _ in range(opts.num_of_runs):

        # otherwise generate data set (new for each run for the small datasets)
        if not opts.load_dataset and not opts.zero_shot:
            data_set = dataset.DataSet(opts.dimensions,
                                        game_size=opts.game_size,
                                        device=opts.device)
            # prepare folders for saving
            if not opts.context_unaware:
                # create subfolder if necessary
                opts.save_path = os.path.join(opts.path, folder_name, 'standard')
                if not os.path.exists(opts.save_path) and opts.save:
                    os.makedirs(opts.save_path)
            elif opts.context_unaware:
                # create subfolder if necessary
                opts.save_path = os.path.join(opts.path, folder_name, 'context_unaware')
                if not os.path.exists(opts.save_path) and opts.save:
                    os.makedirs(opts.save_path)                   

        # zero-shot                
        if opts.zero_shot:
            # either the zero-shot test condition is given (with pre-generated dataset)
            if opts.zero_shot_test is not None:
                # create subfolder if necessary
                opts.save_path = os.path.join(opts.path, folder_name, 'zero_shot', opts.zero_shot_test)
                if not os.path.exists(opts.save_path) and opts.save:
                    os.makedirs(opts.save_path)
                if not opts.load_dataset:
                    data_set = dataset.DataSet(opts.dimensions,
                                                game_size=opts.game_size,
                                                device=opts.device,
                                                zero_shot=True,
                                                zero_shot_test=opts.zero_shot_test)
            # or both test conditions are generated        
            else:
                # implement two zero-shot conditions: test on most generic vs. test on most specific dataset
                for cond in ['generic', 'specific']:
                    print("Zero-shot condition:", cond)
                    # create subfolder if necessary
                    opts.save_path = os.path.join(opts.path, folder_name, 'zero_shot', cond)
                    if not os.path.exists(opts.save_path) and opts.save:
                        os.makedirs(opts.save_path)
                    data_set = dataset.DataSet(opts.dimensions,
                                                game_size=opts.game_size,
                                                device=opts.device,
                                                zero_shot=True,
                                                zero_shot_test=cond)
                    train(opts, data_set, verbose_callbacks=False)

        if opts.zero_shot_test != None or not opts.zero_shot:
            train(opts, data_set, verbose_callbacks=False)
        


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
