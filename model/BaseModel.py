import torch
import torch.nn as nn
from torch.nn import functional as F

class BaseModel(nn.Module):
    def __init__(self, embedding_size, data_size, create_embeddings=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_users = data_size[0]
        self.num_items = data_size[1]
        self.num_words = data_size[2]
        if create_embeddings:
            # initialize features
            self.users_feature = nn.Parameter(
                torch.FloatTensor(self.num_users, self.embedding_size))
            nn.init.xavier_normal_(self.users_feature)
            self.items_feature = nn.Parameter(
                torch.FloatTensor(self.num_items, self.embedding_size))
            nn.init.xavier_normal_(self.items_feature)
            self.words_feature = nn.Parameter(
                torch.FloatTensor(self.num_words, self.embedding_size))
            nn.init.xavier_normal_(self.words_feature)

    def propagate(self, *args, **kwargs):
        '''
        raw embeddings -> embeddings for predicting
        return (user's, item's)
        '''
        raise NotImplementedError

    def predict(self, users_feature, items_feature, *args, **kwargs):
        '''
        embeddings of targets for predicting -> scores
        return scores
        '''
        raise NotImplementedError

    def regularize(self, users_feature, items_feature, *args, **kwargs):
        '''
        embeddings of targets for predicting -> loss regularization(default: MSE Loss...)
        '''
        regularizer = F.mse_loss(users_feature, torch.zeros_like(users_feature), reduction='sum') + F.mse_loss(
            items_feature, torch.zeros_like(items_feature), reduction='sum')
        return regularizer
