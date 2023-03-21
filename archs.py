import torch
import torch.nn as nn
import torch.nn.functional as F


# try out different senders:
# 1) a sender who only sees the targets
# 2) a sender who receives the objects in random order and a vector of labels indicating which are the targets
# 3) a sender who computes prototype embeddings over targets and distractors
# optionally 4) a sender who receives targets first, then distractors and is thus implicitly informed 
# about which are the targets (used in Lazaridou et al. 2017)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features, n_targets):
        super(Sender, self).__init__()
        # n_hidden = 256
        # n_features = 9 (for 3,3,3 dataset)
        # n_objects = 10
        #self.n_features = n_features
        self.n_targets = n_targets
        print(n_hidden)
        print(n_features)
        print(n_targets)
        self.fc1 = nn.Linear(n_features, n_hidden)      # embed input features
        self.fc2 = nn.Linear(n_targets, n_hidden)       # embed objects
        self.fc3 = nn.Linear(2 * n_hidden, n_hidden)    # merge embeddings
        self.fc4 = nn.Linear(n_features * n_targets, n_hidden)

    def forward(self, x, aux_input=None):
        print("input to sender")
        print(x.shape) # [32, 20, 9]
        # input shape: [batch_size, game_size*2, nr_features]
        # embed target features only
        #target_features = x[:,:self.n_targets]
        #print(target_features.shape) # [32, 10, 9]
        #target_feature_embedding = F.relu(self.fc1(target_features))
        #print(target_feature_embedding.shape) # shape: [32, 10, 256]
        # error when passing to GRU wrapper: hidden0 has inconsistent hidden_size: got 10, expected 256 
        #print(target_feature_embedding)
        #return target_feature_embedding

        # emergent-generalization
        batch_size = x.shape[0]
        n_obj = x.shape[1]
        n_features = x.shape[2]
        n_targets = int(n_obj/2)
        #x_flat = x.view(batch_size * n_obj, n_features)
        #print(x_flat.shape)
        # uses MLP to encode features (with at least 2 layers)

        # combination:
        targets = x[:, :self.n_targets]
        targets_flat = targets.reshape(batch_size, n_targets * n_features)
        print(targets_flat.shape) # [32, 90]
        target_feature_embedding_flat = F.relu(self.fc4(targets_flat))
        print(target_feature_embedding_flat.shape) # [32, 256]
        #target_feature_embedding = target_feature_embedding_flat.unsqueeze(1).view(batch_size, n_targets, -1)
        #print(target_feature_embedding.shape)
        return target_feature_embedding_flat
        



#class Sender(nn.Module):
#    def __init__(self, n_hidden, n_features, n_intentions):
#        super(Sender, self).__init__()
#        # n_hidden = 256
#        # n_features = 9
#        # n_intentions = 3
#        self.n_features = n_features
#        self.fc1 = nn.Linear(n_features, n_hidden)      # embed input features
#        self.fc2 = nn.Linear(n_intentions, n_hidden)      # embed intentions
#        self.fc3 = nn.Linear(2 * n_hidden, n_hidden)    # linear layer to merge embeddings

    #def forward(self, x, aux_input=None):
    #    features = x[:, 0:self.n_features]
    #    intentions = x[:, self.n_features:]
    #    feature_embedding = F.relu(self.fc1(features))
    #    intention_embedding = F.relu(self.fc2(intentions))
    #    joint_embedding = torch.cat([feature_embedding, intention_embedding], dim=1)
    #    return self.fc3(joint_embedding).tanh()


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)      # embed input features

    def forward(self, x, features, aux_input=None):
        print("input to receiver")
        print(x.shape) # [32, 256]
        feature_embeddings = self.fc1(features).tanh()
        energies = torch.matmul(feature_embeddings, torch.unsqueeze(x, dim=-1))
        print(energies.squeeze().shape)
        return energies.squeeze()
