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

class CopySpeaker(nn.Module):
    def __init__(
        self,
        feat_model,
        dropout=0.5,
        prototype="average", # average is the only implemented so far
    ):
        super().__init__()
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.emb_size = 2 * self.feat_size
        self.dropout = nn.Dropout(p=dropout)

        self.prototype = prototype

    def embed_features(self, feats, targets=None):
        """
        Prototype to embed positive and negative examples of concept
        """
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        n_features = feats.shape[2]
        n_targets = int(n_obj/2)

        # get target objects:
        #targets = feats[:, :n_targets]
        #targets = torch.zeros(n_obj)
        #targets[:n_targets] = 1
        #print(targets)
        
        feats_flat = feats.view(batch_size * n_obj, *rest)

        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        #print("feats_emb shape", feats_emb.shape)
        targets = feats_emb[:, :n_targets]
        distractors = feats_emb[:, n_targets:]

        if targets is None:
            feats_emb_dropout = self.dropout(feats_emb)
            return feats_emb_dropout
        else:
            prototypes = self.form_prototypes(feats_emb, targets)
            prototypes_dropout = self.dropout(prototypes)

            return prototypes_dropout

    def form_prototypes(self, feats_emb, targets):
        if self.prototype == "average":
            return self._form_average_prototypes(feats_emb, targets)
        elif self.prototype == "transformer":
            raise NotImplementedError("Is implemented in Mu & Goodman, but not yet here.")
        else:
            raise RuntimeError(f"Unknown prototype {self.prototype}")

    def _form_average_prototypes(self, feats_emb, targets):
        """
        Given embedded features and targets, form into prototypes (i.e. average
        together positive examples, average together negative examples)
        """
        rev_targets = 1 - targets # TODO: maybe take distractors here instead!
        #("feature emb and target shape", feats_emb.shape, targets.shape)
        #print("targets squeezed unsqueezed", targets.shape)
        #pos_proto = (feats_emb * targets.unsqueeze(2)).sum(1)
        #neg_proto = (feats_emb * rev_targets.unsqueeze(2)).sum(1)
        pos_proto = (targets).sum(1)
        neg_proto = (rev_targets).sum(1)

        n_pos = targets.sum(1, keepdim=True)
        n_neg = rev_targets.sum(1, keepdim=True)

        # Avoid div by 0 (when n_pos is clamped to min 1, pos_proto is all 0s
        # anyways)
        n_pos = torch.clamp(n_pos, min=1)
        n_neg = torch.clamp(n_neg, min=1)

        n_pos = 10
        n_neg = 10

        # Divide by sums (avoid div by 0 error)
        pos_proto = pos_proto / n_pos
        neg_proto = neg_proto / n_neg

        ft_concat = torch.cat([pos_proto, neg_proto], 1)
        #print("ftconcat", ft_concat.shape)

        return ft_concat

    def forward(self, feats, targets):
        """
        Pass through entire model hidden state
        """
        return self.embed_features(feats, targets)


class Speaker(CopySpeaker):
    def __init__(
        self, feat_model, embedding_module, tau, hidden_size, **kwargs # hidden_size=100 changed to 1024, tau = 1.0
    ):
        super().__init__(feat_model, **kwargs)

        self.embedding_dim = embedding_module.embedding_dim
        self.embedding = embedding_module
        self.vocab_size = embedding_module.num_embeddings
        self.hidden_size = hidden_size
        self.tau = tau

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        # 2 * feat_size - one for positive prototype, one for negative
        #print("linear layer", 2 * self.feat_size, self.hidden_size)
        self.init_h = nn.Linear(2 * self.feat_size, self.hidden_size)
        self.bilinear = nn.Linear(self.hidden_size, self.feat_size, bias=False)

    def forward(self, feats, targets, **kwargs):
        """Sample from image features"""
        feats_emb = self.embed_features(feats, targets)
        #print("feats_emb shape", feats_emb.shape)
        # initialize hidden states using image features
        states = self.init_h(feats_emb)
        #print("states", states.shape)

        return states
        

class CopyListener(nn.Module):
    def __init__(self, feat_model, message_size=100, dropout=0.2):
        super().__init__()

        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.message_size = message_size

        if self.message_size is None:
            self.bilinear = nn.Linear(self.feat_size, 1, bias=False)
        else:
            self.bilinear = nn.Linear(self.message_size, self.feat_size, bias=False)

    def embed_features(self, feats):
        #print("listener", feats.shape)
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)
        #print("listener", feats_flat.shape)
        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        feats_emb = self.dropout(feats_emb)
        #print("listener", feats_emb.shape)
        return feats_emb

    def compare(self, feats_emb, message_enc):
        """
        Compute dot products
        """
        scores = torch.einsum("ijh,ih->ij", (feats_emb, message_enc))
        return scores

    def forward(self, feats, message):
        # Embed features
        feats_emb = self.embed_features(feats)

        # Embed message
        if self.message_size is None:
            return self.bilinear(feats_emb).squeeze(2)
        else:
            message_bilinear = self.bilinear(message)

            return self.compare(feats_emb, message_bilinear)

    def reset_parameters(self):
        self.feat_model.reset_parameters()
        self.bilinear.reset_parameters()


class Listener(CopyListener):
    def __init__(self, feat_model, embedding_module, **kwargs):
        super().__init__(feat_model, **kwargs)

        self.embedding = embedding_module
        self.lang_model = rnn.RNNEncoder(self.embedding, hidden_size=self.message_size)
        #self.vocab_size = embedding_module.num_embeddings

    def forward(self, lang, feats, lang_length):
        # Embed features
        #print("listener forward call", feats.shape, lang.shape)
        feats_emb = self.embed_features(feats)

        # Embed language
        #lang_emb = self.lang_model(lang, lang_length)

        # Bilinear term: lang embedding space -> feature embedding space
        #lang_bilinear = self.bilinear(lang)

        return self.compare(feats_emb, lang) # lang_bilinear
        #dots = torch.matmul(feats_emb, torch.unsqueeze(lang, dim=-1))
        #return dots.squeeze()

    def reset_parameters(self):
        super().reset_parameters()
        self.embedding.reset_parameters()
        self.lang_model.reset_parameters()
