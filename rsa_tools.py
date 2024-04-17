import itertools
import torch


def get_utterances(vocab_size, max_length, interaction_path=None):
    if interaction_path:
        print(f'Loading utterances from {interaction_path}')
        utterances = get_unique_utterances(interaction_path)
    else:
        print(f'Generating utterances with vocab size {vocab_size} and max length {max_length}')
        utterances = generate_utterances(vocab_size, max_length)

    print(f'Shape of utterances: {utterances.shape}')

    # Convert to one-hot encoding
    utterances = torch.nn.functional.one_hot(utterances, num_classes=vocab_size).float()
    return utterances


def generate_utterances(vocab_size, max_length):
    """
    Generate all possible utterances using vocab_size and max_length
    """
    all_possible_utterances = []
    # Max length plus one for EOS symbol
    total_length = max_length + 1
    # Loop through each possible utterance length (1 to max_length)
    for length in range(1, max_length + 1):
        # Generate all combinations for current length
        for combination in itertools.product(range(1, vocab_size), repeat=length):
            # Append EOS
            utterance = list(combination) + [0]

            # Apply padding to the right for shorter utterances
            padding = [0] * (total_length - len(utterance))
            utterance = utterance + padding

            all_possible_utterances.append(utterance)
    return torch.tensor(all_possible_utterances)


def get_unique_utterances(interaction_path):
    """
    Get unique utterances from interaction data
    """
    interaction_train = torch.load(interaction_path)
    messages_train = interaction_train.message.argmax(dim=-1)

    interaction_val = torch.load(interaction_path.replace('train', 'validation'))
    messages_val = interaction_val.message.argmax(dim=-1)

    messages = torch.cat((messages_train, messages_val), dim=0)
    messages = [msg.tolist() for msg in messages]
    unique_messages = [list(x) for x in set(tuple(x) for x in messages)]

    return torch.tensor(unique_messages)
