import pickle
from egg.core.language_analysis import Disent
from utils.analysis_from_interaction import *

datasets = ("(3,16)",)
n_attributes = (3,)
n_values = (16,)
n_epochs = 300
paths = [f"results/vague_ds_results/results/{d}_game_size_10_vsf_3/" for d in datasets]

context_unaware = False  # whether original or context_unaware simulations are evaluated
length_cost = False
if context_unaware:
    setting = "context_unaware"
elif length_cost:
    setting = "length_cost_001"
else:
    setting = "standard"

# use Disent callback from egg

for d in range(len(datasets)):

    path = paths[d]
    dim = [n_values[d]] * n_attributes[d]
    n_features = n_attributes[d] * n_values[d]
    vs_factor = int(path[-2])
    vocab_size = (n_values[d] + 1) * vs_factor + 1

    print("data set", dim)

    for run in range(3):

        posdis_bosdis = {}

        path_to_run = paths[d] + "/" + str(setting) + "/" + str(run) + "/"
        path_to_interaction_train = (
            path_to_run
            + "interactions/train/epoch_"
            + str(n_epochs)
            + "/interaction_gpu0"
        )
        interaction = torch.load(path_to_interaction_train)

        messages = interaction.message.argmax(dim=-1)
        sender_input = interaction.sender_input
        n_targets = int(sender_input.shape[1] / 2)
        # get target objects and fixed vectors to re-construct concepts
        target_objects = sender_input[:, :n_targets]
        target_objects = k_hot_to_attributes(target_objects, n_values[d])
        # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
        (objects, fixed) = retrieve_concepts_sampling(target_objects)
        # add one such that zero becomes an empty attribute for the calculation (_)
        objects = objects + 1
        concepts = torch.from_numpy(objects * (np.array(fixed)))

        # concrete/specific concepts: where all attributes are fixed
        concepts_specific = torch.tensor(
            objects[torch.sum(torch.from_numpy(fixed), dim=1) == n_attributes[d]]
        )
        messages_specific = messages[
            torch.sum(torch.from_numpy(fixed), dim=1) == n_attributes[d]
        ]

        # generic concepts: where only one attribute is fixed
        concepts_generic = torch.tensor(
            objects[torch.sum(torch.from_numpy(fixed), dim=1) == 1]
        )
        messages_generic = messages[torch.sum(torch.from_numpy(fixed), dim=1) == 1]

        posdis_specific = Disent.posdis(concepts_specific, messages_specific)
        bosdis_specific = Disent.bosdis(
            concepts_specific, messages_specific, vocab_size
        )

        posdis_generic = Disent.posdis(concepts_generic, messages_generic)
        bosdis_generic = Disent.bosdis(concepts_generic, messages_generic, vocab_size)

        posdis = Disent.posdis(torch.from_numpy(objects), messages)
        bosdis = Disent.bosdis(torch.from_numpy(objects), messages, vocab_size)

        posdis_bosdis["posdis_specific"] = posdis_specific
        posdis_bosdis["bosdis_specific"] = bosdis_specific
        posdis_bosdis["posdis_generic"] = posdis_generic
        posdis_bosdis["bosdis_generic"] = bosdis_generic
        posdis_bosdis["posdis"] = posdis
        posdis_bosdis["bosdis"] = bosdis

        print(posdis_bosdis)

        pickle.dump(posdis_bosdis, open(path_to_run + "posdis_bosdis.pkl", "wb"))
