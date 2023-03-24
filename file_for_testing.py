
def main():
    path_to_interaction = 'results/(4,4)_game_size_10_vsf_3/standard/0/interactions/train/epoch_100/interaction_gpu0'
    interaction = torch.load(path_to_interaction)
    n_dims = 4
    n_values = 4
    scores = information_scores(interaction, n_dims, n_values, normalizer="arithmetic")
    #pickle.dump(scores, open( + 'entropy_scores.pkl', 'wb'))


if __name__ == "__main__":

    import utils
    import torch
    from utils.analysis_from_interaction import *
    main()



