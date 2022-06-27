import torch
import torch.nn as nn
from torch.nn import functional as F

class BaseModel(nn.Module):
    def __init__(self, embedding_size, data_size, create_embeddings=True):
        super().__init__()
        self.embedding_size = embedding_size
        data_size_A, data_size_B = data_size
        self.num_users_A = data_size_A[0]
        self.num_items_A = data_size_A[1]
        self.num_words_A = data_size_A[2]
        self.num_users_B = data_size_B[0]
        self.num_items_B = data_size_B[1]
        self.num_words_B = data_size_B[2]
        self.ind_A = torch.tensor([i for i in range(self.num_words_A)])
        self.ind_B = torch.tensor([i for i in range(self.num_words_B)])
        if create_embeddings:
            # initialize features_A
            self.users_feature_A = nn.Parameter(
                torch.FloatTensor(self.num_users_A, self.embedding_size))
            nn.init.xavier_normal_(self.users_feature_A)
            self.items_feature_A = nn.Parameter(
                torch.FloatTensor(self.num_items_A, self.embedding_size))
            nn.init.xavier_normal_(self.items_feature_A)
            self.words_feature_A = nn.Parameter(
                torch.FloatTensor(self.num_words_A, self.embedding_size))
            nn.init.xavier_normal_(self.words_feature_A)
            # initialize features_B
            self.users_feature_B = nn.Parameter(
                torch.FloatTensor(self.num_users_B, self.embedding_size))
            nn.init.xavier_normal_(self.users_feature_B)
            self.items_feature_B = nn.Parameter(
                torch.FloatTensor(self.num_items_B, self.embedding_size))
            nn.init.xavier_normal_(self.items_feature_B)
            self.words_feature_B = nn.Parameter(
                torch.FloatTensor(self.num_words_B, self.embedding_size))
            nn.init.xavier_normal_(self.words_feature_B)

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
