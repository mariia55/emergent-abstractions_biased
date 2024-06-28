# Created by Eosandra Grund
# contains analysis functions to analyse results saved as interactions
# average message length
# 


# ZLA significance score by Rita et al. 
# https://github.com/MathieuRita/Lazimpa egg.zoo.channel.test ???
# ********************

import torch
from language_analysis_local import MessageLengthHierarchical
from utils.analysis_from_interaction import *

def mean_message_length_from_interaction(interaction):
    """ Calculates the average message length, with only accounting for each individual message one time. """
    messages = interaction.message.argmax(dim=-1)
    unique_messages = torch.unique(messages,dim=0)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length

def mean_weighted_message_length_from_interaction(interaction):
    """ average length of the messages weighted by their generation frequency (how often does the agent generate this message)"""
    messages = interaction.message.argmax(dim=-1)
    message_length = MessageLengthHierarchical.compute_message_length(messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length

def mean_weighted_message_length(length,frequency):
    """ calculates the average length of the messages wieghted by their generation frequency from length and frequency. """
    return torch.sum(((length * frequency)/torch.sum(frequency)).float())


def ZLA_significance_score(interaction,num_permutations = 1000):
    """ Calculates to which degree mean_weighted_message_length is lower than a random permutation of its frequency length mapping """

    # get message frequencies and message_length
    messages = interaction.message.argmax(dim=-1)
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
                  'Pzla':pZLA}
    return score_dict

# positional encoding by Rita et al. 
# *************************************

def positional_encoding():
    """ """
    pass

def effective_length():
    """ """
    pass

def information_density():
    """ """
    pass

# other functions I might need
#**************************

def load_interaction(path,setting,nr=0,n_epochs=300):
    """ loads an interaction given the path, setting, nr and how many epochs where done (Path before setting)"""
    # select first run
    path_to_run = path + '/' + str(setting) +'/' + str(nr) + '/'
    path_to_interaction_train = (path_to_run + 'interactions/train/epoch_' + str(n_epochs) + '/interaction_gpu0')
    path_to_interaction_val = (path_to_run + 'interactions/validation/epoch_' + str(n_epochs) + '/interaction_gpu0')
    interaction = torch.load(path_to_interaction_train)
    print(path_to_interaction_train)
    return interaction

def retrieve_messages(interaction,as_list = True):
    messages = interaction.message.argmax(dim=-1)
    if not as_list:
        return messages
    return [msg.tolist() for msg in messages]

def retrieve_concepts_context(interaction,n_values):
    """ retrieves Concepts and context conditions from an interaction"""
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