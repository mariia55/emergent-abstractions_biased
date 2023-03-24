# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import argparse
import torch
#import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.core.language_analysis import TopographicSimilarity
# copy language_analysis_local from hierarchical_reference_game?
from language_analysis_local import *
import os
import pickle

import dataset
# TBI: archs
from archs import Sender, Receiver


SPLIT = (0.6, 0.2, 0.2)
# SPLIT_ZERO_SHOT = (0.75, 0.25) # split for train and val only?


def get_params(params):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_dataset', type=str, default=None,
                        help='If provided that data set is loaded. Datasets can be generated with pickle.ds'
                            'This makes sense if running several runs with the exact same dataset.')
    parser.add_argument('--dimensions', nargs='+', type=int)
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
    
    args = core.init(parser, params)

    return args


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Loss needs to be defined for gumbel softmax relaxation.
    For a discriminative game, accuracy is computed by comparing the index with highest score in Receiver
    output (a distribution of unnormalized probabilities over target positions) and the corresponding 
    label read from input, indicating the ground-truth position of the target.
        receiver_output: Tensor of shape [batch_size, n_objects]
        labels: Tensor of shape [batch_size, n_objects]
    """
    def _many_hot_encoding(n_objects, input_list):
        """
	    Outputs a binary one dim vector
	    """
        output = torch.zeros([n_objects])
        for i in range(n_objects):
            for index in input_list:
                if i == index:
                    output[i] = 1

        return output
    
    batch_size = receiver_output.shape[0]
    n_objects = receiver_output.shape[1]
    # Can't simply use argmax because I've got 10 target labels (out of 20), not just 1.
    # So I use topk and calculate the topk indices outputted by the receiver
    _topk_values, topk_indices = receiver_output.topk(k=int(n_objects/2), dim=1)
    # forming a many-hot-encoding of the (sorted) topk indices to match the shape of the ground-truth labels
    sorted, _ = torch.sort(topk_indices)
    receiver_pred = torch.cat([_many_hot_encoding(n_objects, label) for label in sorted]).reshape(batch_size,n_objects).to(device='cuda')
    # comparing receiver predictions for all objects with ground-truth labels
    acc_all_objects = (receiver_pred == labels).detach().float() # shape [batch_size, n_objects]
    # NOTE: accuracy shape needs to be [32] to fit with egg code !!!
    # This means that accuracy is 1 only when all objects are classified correctly 
    # (which makes it a harder task than a simple referential game with one target only).
    # re-calculating accuracy over all objects:
    acc = list()
    all_correct = torch.ones(n_objects).to(device='cuda')
    for row in acc_all_objects:
        if torch.equal(row, all_correct):
            acc.append(1)
        else:
            acc.append(0)
    acc = torch.Tensor(acc).to(device='cuda')

    # from EGG: similarly, the loss computes cross-entropy between the Receiver-produced 
    # target-position probability distribution and the labels
    # TODO: sanity check the loss calculation
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {'acc': acc}


def train(opts, datasets, verbose_callbacks=True): # TODO: fix and set to True
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
    sender = Sender(opts.hidden_size, sum(dimensions), opts.game_size)
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
            TopographicSimilarityHierarchical(dimensions, is_gumbel=True,
                                              save_path=opts.save_path, save_epoch=save_epoch),
            MessageLengthHierarchical(len(dimensions),
                                      print_train=True, print_test=True, is_gumbel=True,
                                      save_path=opts.save_path, save_epoch=save_epoch)
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

    if not opts.zero_shot:
        # evaluate accuracy and topsim where all attributes are relevant
        max_same_indices = torch.where(torch.sum(interaction.sender_input[:, -len(dimensions):], axis=1) == 0)[0]
        acc = torch.mean(interaction.aux['acc'][max_same_indices]).item()
        sender_input = interaction.sender_input[max_same_indices]
        messages = interaction.message[max_same_indices]
        messages = messages.argmax(dim=-1)
        messages = [msg.tolist() for msg in messages]
        #TODO: needs to be adapted before use
        #sender_input_hierarchical = encode_input_for_topsim_hierarchical(sender_input, dimensions)
        #topsim_hierarchical = TopographicSimilarity.compute_topsim(sender_input_hierarchical,
                                                                   #messages,
                                                                   #meaning_distance_fn="cosine",
                                                                   #message_distance_fn="edit")
        max_nsame_dict = dict()
        max_nsame_dict['acc'] = acc
        # TODO: this probably needs to be adapted
        #max_nsame_dict['topsim_hierarchical'] = topsim_hierarchical
        print("maximal #same eval", max_nsame_dict)

        if opts.save:
            pickle.dump(max_nsame_dict, open(opts.save_path + '/max_nsame_eval.pkl', 'wb'))


def main(params):
    """
    Dealing with parameters and loading dataset. Copied from hierarchical_reference_game and adapted.
    """
    opts = get_params(params)

    # has to be executed in Project directory for consistency
    assert os.path.split(os.getcwd())[-1] == 'emergent-abstractions'

    data_set_name = '(' + str(len(opts.dimensions)) + ',' + str(opts.dimensions[0]) + ')'
    folder_name = (data_set_name + '_game_size_' + str(opts.game_size) 
                    + '_vsf_' + str(opts.vocab_size_factor))
    folder_name = os.path.join("results", folder_name)

    # if name of precreated data set is given, load dataset
    if opts.load_dataset:
        data_set = torch.load('data/' + opts.load_dataset)
        print('data loaded from: ' + 'data/' + opts.load_dataset)

    for _ in range(opts.num_of_runs):

        # otherwise generate data set
        if not opts.load_dataset:
            data_set = dataset.DataSet(opts.dimensions,
                                        game_size=opts.game_size)
        if opts.zero_shot:
            raise NotImplementedError
            ## create subfolder if necessary
            #opts.save_path = os.path.join(folder_name, 'zero_shot')
            #if not os.path.exists(opts.save_path) and opts.save:
            #    os.makedirs(opts.save_path)
            #train(opts, item_set.get_zero_shot_datasets(SPLIT_ZERO_SHOT), verbose_callbacks=False)

        else:
            # create subfolder if necessary
            opts.save_path = os.path.join(folder_name, 'standard')
            if not os.path.exists(opts.save_path) and opts.save:
                os.makedirs(opts.save_path)
            train(opts, data_set.get_datasets(split_ratio=SPLIT), verbose_callbacks=True) # TODO: fix and set to True


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
