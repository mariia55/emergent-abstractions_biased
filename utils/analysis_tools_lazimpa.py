# Created by Eosandra Grund
# contains analysis functions to analyse results saved as interactions
# average message length
# 


# ZLA significance score as described by Rita et al. (2020)
# ********************

import torch
from language_analysis_local import MessageLengthHierarchical
from utils.analysis_from_interaction import *
import egg.core as core
from archs import Sender, Receiver
from models_lazimpa import LazImpaSenderReceiverRnnGS
import os
import train
import pickle

def mean_message_length_from_interaction(interaction):
    """ Calculates the average message length, with only accounting for each individual message one time. 
    
    :param interaction: interaction (EGG class)
    """
    messages = interaction.message.argmax(dim=-1)
    unique_messages = torch.unique(messages,dim=0)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length

def mean_weighted_message_length_from_interaction(interaction):
    """ average length of the messages weighted by their generation frequency (how often does the agent generate this message)
    
    :param interaction: interaction (EGG class)
    """
    messages = interaction.message.argmax(dim=-1)
    message_length = MessageLengthHierarchical.compute_message_length(messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length

def mean_weighted_message_length(length,frequency): # TODO analyse context x concept counts and if it matches this calculation but should
    """ calculates the average length of the messages wieghted by their generation frequency from length and frequency. 
    
    :param length: torch.Tensor with shape = (n,) contains the length of each message
    :param frequency: torch.Tensor with shape = (n,) contains the frequency of each message
    """
    return torch.sum(((length * frequency)/torch.sum(frequency)).float())


def ZLA_significance_score(interaction,num_permutations = 1000, remove_after_eos = True):
    """ Calculates to which degree mean_weighted_message_length is lower than a random permutation of its frequency length mapping 
    
    :param interaction: interaction (EGG class)
    :param num_permutations: int
    :param remove_after_eos: bool (should be true, as two messages should be the same if the only difference is after eos)
    """

    # get message frequencies and message_length
    messages = retrieve_messages(interaction,as_list=False,remove_after_eos=remove_after_eos,eos_token=0.0)
    unique_messages, frequencies = torch.unique(messages,dim=0,return_counts=True)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)

    L_type = torch.mean(message_length.float()) # mean message length
    original_L_token = mean_weighted_message_length_from_interaction(interaction)

    # random permutations of frequency-length mapping, then calulate their L_token
    permuted_L_tokens = []
    for _ in range(num_permutations):
        permuted_indices = torch.randperm(message_length.shape[0])
        permuted_lengths = message_length[permuted_indices]
        permuted_L_token = mean_weighted_message_length(permuted_lengths,frequencies)
        permuted_L_tokens.append(permuted_L_token)

    # calculate p_value
    pZLA = (torch.tensor(permuted_L_tokens) <= original_L_token).float().mean()

    score_dict = {'mean_message_length':L_type,
                  'mean_weighted_message_length':original_L_token,
                  'p_zla':pZLA}
    return score_dict

# positional encoding as described by Rita et al. (2020)
# *************************************

def symbol_informativeness(listener,interaction,eos_token = 0.0):
    """ Calculates for each symbol if it is informative or not by switching its value with a random other symbols value for the same position (except eos) and observing if the prediction (until eos) of the listener changes. 
    
    :param listener:
    :param interaction: interaction (EGG class)
    :param eos_token: float

    Example switch for position 1 in message with message length 2, switch max = 2 with switch_index = 1 (here only option as eos and max are excluded)
    original_message = tensor([[0.2,0.3,0.5],[0.3,0.5,0.2]]) -> message = [2,1]
    manipulated_message = tensor([[0.2,0.5,0.3],[0.3,0.5,0.2]]) -> message = [1,1]
    """

    messages = messages_without_after_eos(interaction.message,eos_token)
    num_messages, message_length, vocab_size = messages.shape

    # values to be switched
    max_values, original_messages = messages.max(dim=-1)

    eos_mask_total = eos_token != original_messages

    # only values before eos, to switch
    eos_excluded_cumulative = torch.cat([ eos_mask_total[:,:i+1].sum(dim=1).unsqueeze(1) == i+1 for i in range(message_length) ],dim=1)

    # until eos, evaluating changes
    eos_included_cumulative = torch.cat([torch.ones_like(eos_excluded_cumulative[:,-1]).unsqueeze(1).bool(), eos_excluded_cumulative[:,:-1]],dim=1)

    # Prediction for original messages, exclude prediction for values after eos
    prediction = (listener(messages,interaction.receiver_input) > 0).float()
    empty_prediction = torch.zeros_like(prediction)
    prediction = torch.where(eos_included_cumulative.unsqueeze(-1),prediction,empty_prediction)

    Lambda_m_k_list = []

    for i in range(message_length-1): # index = -1 is always eos and not changed. 

        # pick random new symbol index to switch with
        switch_index = torch.randint(1,vocab_size,(num_messages,))

        # make sure switch_index != original_messages
        same_indices = (switch_index == original_messages[:,i])
        while same_indices.any():
            switch_index[same_indices] = torch.randint(1, vocab_size, (same_indices.sum().item(),))
            same_indices = (switch_index == original_messages[:,i])

        # values to switch with
        new_value = torch.gather(messages[:,i,:],-1,switch_index.unsqueeze(-1)).squeeze()

        messages_manipulated = messages.clone()

        # if eosed, take old value in so no switch
        new_value_without_eos = torch.where(eos_excluded_cumulative[:,i].squeeze(), new_value,max_values[:,i])
        max_value_without_eos = torch.where(eos_excluded_cumulative[:,i].squeeze(),max_values[:,i],new_value)

        # switch value i of all messages to new symbol
        messages_manipulated[:,i].scatter_(-1,original_messages[:,i].unsqueeze(-1),new_value_without_eos.unsqueeze(-1))
        messages_manipulated[:,i].scatter_(-1,switch_index.unsqueeze(-1),max_value_without_eos.unsqueeze(-1))

        prediction_manipulated = (listener(messages_manipulated,interaction.receiver_input) > 0).float() # bigger 0 means thinks its a concept object
        prediction_manipulated = torch.where(eos_included_cumulative.unsqueeze(-1),prediction_manipulated,empty_prediction)

        # compare whether same classification not same values
        # only include predictions for symbols that are not eosed yet.
        Lambda_m_k_position = (prediction != prediction_manipulated).sum(dim=(1,2)).unsqueeze(1).bool() # (num_messages, 1 )
        Lambda_m_k_list.append(Lambda_m_k_position)

    Lambda_m_k = torch.cat(Lambda_m_k_list,dim=1)

    return Lambda_m_k, eos_excluded_cumulative[:,:-1] # both should be the same shape

def positional_encoding(Lambda_m_k,eos_cumulative):
    """ Calculates a vector Lambda_dot_k, that contains the proportions of informative symbols for each position in messages. Lambda_m_k. Excludes eos 
    
    :param Lambda_m_k: torch.Tensor with shape (num_messages,message_lenght - 1)
    :param eos_cumulative: torch.Tensor with shape (num_messages,message_lenght - 1)
    """
    nr_non_eos_per_position = eos_cumulative.sum(dim=0)
    nr_informative_per_position = Lambda_m_k.sum(dim=0)
    return nr_informative_per_position / nr_non_eos_per_position

def effective_length(Lambda_m_k,eos_cumulative):
    """ Measures the mean number of informative symbols by message.
    
    :param Lambda_m_k: torch.Tensor with shape (num_messages,message_lenght - 1)
    :param eos_cumulative: torch.Tensor with shape (num_messages,message_lenght - 1)
    """
    nr_informative_per_message = Lambda_m_k.sum(dim=1)
    nr_informative_per_message_mean = nr_informative_per_message.float().mean()
    return nr_informative_per_message_mean
    
def information_density(Lambda_m_k,eos_cumulative):
    """ Measures the fraction of informative symbols in a language. 
    
    :param Lambda_m_k: torch.Tensor with shape (num_messages,message_lenght - 1)
    :param eos_cumulative: torch.Tensor with shape (num_messages,message_lenght - 1)
    """
    nr_non_eos = eos_cumulative.sum()
    nr_informative = Lambda_m_k.sum()
    return nr_informative / nr_non_eos

def information_analysis(listener,interaction,eos_token = 0.0):
    """ Calculates positional_encoding, effective_length and information_density. by messages in language, not individual messages
    
    :param listener:
    :param interaction: interaction
    :param eos_token: float
    """

    Lambda_m_k, eos_mask = symbol_informativeness(listener,interaction,eos_token)

    Lambda_dot_k = positional_encoding(Lambda_m_k, eos_mask)
    L_eff = effective_length(Lambda_m_k, eos_mask)
    rho_inf = information_density(Lambda_m_k,eos_mask)

    return_dict = {
        "positional_encoding" : Lambda_dot_k,
        "effective_length" : L_eff,
        "information_density" : rho_inf
    }
    return return_dict

# Other analysis tools
#***************************

def messages_without_after_eos(messages,eos_token = 0.0):
    """ replaces all values eos and after with eos, so that two messages that are the same are acknoledged as so 
    
    :param messages: torch.Tensor:  messages in form of probability distributions (batch,number,vocab_size) 
    :param eos_token: float
    :returns messages: torch.Tensor: shape (batch,number,vocab_size) like param messages, but changed
    """

    original_messages = messages.argmax(dim=-1)

    eos_mask_total = eos_token != original_messages

    # only values before eos including eos, we keep
    eos_excluded_cumulative = torch.cat([ eos_mask_total[:,:i+1].sum(dim=1).unsqueeze(1) == i+1 for i in range(messages.shape[1]) ],dim=1)
    eos_included_cumulative = torch.cat([torch.ones_like(eos_excluded_cumulative[:,-1]).unsqueeze(1).bool(), eos_excluded_cumulative[:,:-1]],dim=1)

    eos_probs = messages[0,-1]
    return torch.where(eos_included_cumulative.unsqueeze(-1),messages,eos_probs)

def percent_wrong_from_interaction(interaction):
    """
    Calculates to what percentage the final prediction of the listener is wrong (how many of the 20 objects are classified wrongly).

    :param interaction: interaction (EGG class)
    :returns percent_wrong: torch.Tensor in shape interaction.messages.shape[0], percentage of final prediction is wrong
    :returns percent_wrong: torch Tensor same shape, if at least one is wrong 1 else 0
    """
    messages = retrieve_messages(interaction,False,remove_after_eos=True)

    # 1. create eos_exact mask
    eos_mask_total = 0.0 != messages
    # only values before eos, to switch
    eos_excluded_cumulative = torch.cat([ eos_mask_total[:,:i+1].sum(dim=1).unsqueeze(-1) == i+1 for i in range(messages.shape[1]) ],dim=-1)
    # until eos, evaluating changes
    eos_included_cumulative = torch.cat([torch.ones_like(eos_excluded_cumulative[:,-1]).unsqueeze(-1).bool(), eos_excluded_cumulative[:,:-1]],dim=-1)
    eos_exact = (eos_included_cumulative != eos_mask_total) # only symbol first eos

    # get scores
    predictions = (interaction.receiver_output > 0).float()
    # remove after eos prediction
    predictions = torch.where(eos_exact.unsqueeze(-1),predictions,interaction.labels.unsqueeze(1))
    score = (predictions != interaction.labels.unsqueeze(1)).float()

    # average over step and objects with 20 being the number ob objects + distractors the listener evaluates
    percent_wrong = ((score.sum(dim=(1,2)) / 20) * 100).int()

    return percent_wrong, percent_wrong.bool().int()

# loader from interaction or other
#**************************

def load_interaction(path,setting,nr=0,n_epochs=300):
    """ loads an interaction given the path, setting, nr and how many epochs where done (Path before setting)
    
    :param path: str
    :param setting: str
    :param nr: int
    :param n_epochs: int
    """
    # select first run
    path_to_run = path + '/' + str(setting) +'/' + str(nr) + '/'
    path_to_interaction_train = (path_to_run + 'interactions/train/epoch_' + str(n_epochs) + '/interaction_gpu0')
    path_to_interaction_val = (path_to_run + 'interactions/validation/epoch_' + str(n_epochs) + '/interaction_gpu0')
    interaction = torch.load(path_to_interaction_train)
    print(path_to_interaction_train)
    return interaction

def retrieve_messages(interaction, as_list = True, remove_after_eos = False, eos_token = 0.0):
    """ retrieves the messages from an interaction 
    
    :param interaction: interaction (EGG class)
    :param as_list: bool
    :param remove_after_eos: bool
    :param eos_token: float
    """
    messages = interaction.message

    if remove_after_eos:
        messages = messages_without_after_eos(messages,eos_token)
    if not as_list:
        return messages.argmax(dim=-1)
    return [msg.tolist() for msg in messages.argmax(dim=-1)]

def retrieve_concepts_context(interaction,n_values):
    """ retrieves Concepts and context conditions from an interaction
    
    :param interaction: interaction (EGG class)
    :param n_values: int
    """
    sender_input = interaction.sender_input
    print(sender_input.shape)
    n_targets = int(sender_input.shape[1]/2)
    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects, all_targets=True)
    concepts = list(zip(objects, fixed))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)
    
    return concepts, context_conds

def load_listener(path, setting, run, n_attributes, n_values, context_unaware, game_size = 10, loss = train.loss, vocab_size_factor = 3, hidden_size = 128,sender_cell='gru',receiver_cell='gru',device='cpu'):
    """ loads the trained listener 
    
    :param path:
    :param setting:
    :param run:
    :param n_attributes:
    :param n_values:
    :param context unaware:
    :param game_size:
    :param loss:
    :param vocab_size_factor:
    :param hidden_size:
    :param sender_cell:
    :param receiver_cell:
    :param device: 'cpu' or 'gpu' to load all listeners on the same device and not on the one they were trained on.
    """
    
    dimensions = list(itertools.repeat(n_values, n_attributes))
    minimum_vocab_size = dimensions[0] + 1
    vocab_size = minimum_vocab_size * vocab_size_factor + 1  # multiply by factor plus add one for eos-symbol
    sender = Sender(hidden_size, sum(dimensions), game_size, context_unaware)
    receiver = Receiver(sum(dimensions), hidden_size)
    sender = core.RnnSenderGS(sender,
                              vocab_size,
                              int(hidden_size / 2),
                              hidden_size,
                              cell=sender_cell,
                              max_len=len(dimensions), # this if not max_mess_len then max_mess_len
                              temperature=2) # value found in analysis_eval.ipynb
    listener = core.RnnReceiverGS(receiver,
                                vocab_size,
                                int(hidden_size / 2),
                                hidden_size,
                                cell=receiver_cell)
    
    game = LazImpaSenderReceiverRnnGS(sender, listener, loss, length_cost=1.0,threshold=1.0) # length_cost and threshold are only important for loss calculation

    #optimizer = torch.optim.Adam([
    #    {'params': game.sender.parameters(), 'lr': opts.learning_rate},
    #    {'params': game.receiver.parameters(), 'lr': opts.learning_rate}
    #])
    
    checkpoint_path = path + '/' + setting + '/' + str(run) + '/final.tar'
    if not os.path.exists(checkpoint_path):
        raise ValueError(
            f"Checkpoint file {checkpoint_path} not found.")
    if device != None:
        checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    else:
        checkpoint = torch.load(checkpoint_path)

    game.load_state_dict(checkpoint[1])

    return listener, checkpoint, game

def load_loss(path, run=0, n_epochs=300,metrics=1):
    """
    loads all losses into a dictionary

    :param: path = path before run
    :run: int
    :n_epochs: int
    :metrics: How many metrics to are to be expected /supposed to be loaded (1 means only accuracy for standard, 4 for lazy or impatience)
    :set_type: str either 'train' oder 'test'
    """

    result_dict = {}
    data = pickle.load(open(path + "/" + str(run) + "/" + 'loss_and_metrics.pkl', 'rb'))

    for s in ['train', 'test']:

        lists = sorted(data['loss_'+s].items())
        _, result_dict[s+'_loss'] = zip(*lists)

        if metrics > 0:
            for m in range(metrics):
                lists = sorted(data['metrics_'+s + str(m)].items())
                _, result_dict[s+'_metric' + str(m)] = zip(*lists)

    return result_dict