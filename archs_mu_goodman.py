import torch
import torch.nn as nn
import torch.nn.functional as F
import rnn

# try out different senders:
# 1) a sender who only sees the targets
# 2) a sender who receives the objects in random order and a vector of labels indicating which are the targets
# 3) a sender who computes prototype embeddings over targets and distractors
# 4) a sender who receives targets first, then distractors and is thus implicitly informed 
# about which are the targets (used in Lazaridou et al. 2017)

"""
Speaker models
"""


def _form_average_prototypes(feats_emb, n_targets):
    """
    Given embedded features and targets, form into prototypes (i.e. average
    together positive examples, average together negative examples)
    - targets is a vector of indices (1 = target, 0 = distractor)
    """
    # XO: In this function you input feats_emb but it is never used; only targets is used.
    # The prototypes of the distractors are not calculated anywhere. Why are they missing? (See your to do)
    # What is 1 - targets supposed to be?

    targets = torch.zeros(n_targets * 2, device='cuda')
    targets[:n_targets] = 1.0

    rev_targets = 1 - targets  # calculate distractor indices

    pos_proto = (feats_emb * targets.unsqueeze(1)).sum(1)
    neg_proto = (feats_emb * rev_targets.unsqueeze(1)).sum(1)

    # feats_emb are the embedded features (32,20,512)
    # first 10 are the targets -> sum to (32,512), one sum per batch
    # pos_proto = feats_emb[:, :n_targets].sum(1)
    # next 10 are the distractors
    # neg_proto = feats_emb[:, n_targets:].sum(1)

    # I always have the same number of targets and distractors
    n_pos = n_targets
    n_neg = n_targets

    # Divide by number of positive and negative examples
    pos_proto = pos_proto / n_pos
    neg_proto = neg_proto / n_neg

    ft_concat = torch.cat([pos_proto, neg_proto], 1)

    return ft_concat


class Speaker(nn.Module):
    def __init__(
            self,
            feat_model,
            dropout=0.5,
            prototype="average",  # average is the only implemented so far
            n_targets=10
    ):
        super().__init__()
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.emb_size = 2 * self.feat_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_targets = n_targets

        self.prototype = prototype

    def embed_features(self, feats):
        """
        Prototype to embed positive and negative examples of concept
        """

        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        n_features = feats.shape[2]

        feats_flat = feats.view(batch_size * n_obj, n_features)

        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)

        prototypes = self.form_prototypes(feats_emb)
        prototypes_dropout = self.dropout(prototypes)

        return prototypes_dropout

    def form_prototypes(self, feats_emb):
        if self.prototype == "average":
            return _form_average_prototypes(feats_emb, self.n_targets)
        elif self.prototype == "transformer":
            raise NotImplementedError("Is implemented in Mu & Goodman, but not yet here.")
        else:
            raise RuntimeError(f"Unknown prototype {self.prototype}")

    def forward(self, feats, _aux_input=None):
        """
        Pass through entire model hidden state
        """
        return self.embed_features(feats)


class Listener(nn.Module):
    def __init__(self, feat_model, dropout=0.2):
        super().__init__()

        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.dropout = nn.Dropout(p=dropout)

    def embed_features(self, feats):
        feats_emb = self.feat_model(feats)

        feats_emb = self.dropout(feats_emb)
        return feats_emb

    def forward(self, x, feats, _aux_input=None):
        # Embed features
        feats_emb = self.embed_features(feats)

        dots = torch.matmul(feats_emb, torch.unsqueeze(x, dim=-1))
        return dots.squeeze()
