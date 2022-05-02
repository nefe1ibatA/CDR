from xml.sax.handler import all_features
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from model.BaseModel import BaseModel
from torch.nn import functional as F
from utils import get_features


def graph_generating(raw_graph, row, col):
    if raw_graph.shape == (row, col):
        graph = sp.bmat([[sp.identity(raw_graph.shape[0]), raw_graph],
                             [raw_graph.T, sp.identity(raw_graph.shape[1])]])
    else:
        raise ValueError(r"raw_graph's shape is wrong")
    return graph

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph



class Model(BaseModel):
    def __init__(self, embedding_size, data_size, raw_graph, num_layers, device, pretrain=None):
        super().__init__(embedding_size, data_size, create_embeddings=True)

        assert isinstance(raw_graph, list)
        ui_graph, uw_graph, iw_graph, ww_graph = raw_graph

        #  deal with weights (uu_graph)
        uw_norm = sp.diags(1 / (np.sqrt((uw_graph.multiply(uw_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ uw_graph
        uu_graph = uw_norm @ uw_norm.T

        #  deal with weights (ii_graph)
        iw_norm = sp.diags(1 / (np.sqrt((iw_graph.multiply(iw_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ iw_graph
        ii_graph = iw_norm @ iw_norm.T

        #  pooling graph (uw)
        user_size = uw_graph.sum(axis=1) + 1e-8
        uw_graph = sp.diags(1/user_size.A.ravel()) @ uw_graph

        #  pooling graph (iw)
        item_size = iw_graph.sum(axis=1) + 1e-8
        iw_graph = sp.diags(1/item_size.A.ravel()) @ iw_graph

        if ww_graph.shape == (self.num_words, self.num_words):
            atom_graph = ww_graph
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
        print('finish generating atom graph')

        if ui_graph.shape == (self.num_users, self.num_items) \
                and uu_graph.shape == (self.num_users, self.num_users) \
                    and ii_graph.shape == (self.num_items, self.num_items):
            # add self-loop
            non_atom_graph = sp.bmat([[uu_graph, ui_graph],
                                 [ui_graph.T, ii_graph]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.non_atom_graph = to_tensor(laplace_transform(non_atom_graph)).to(device)
        print('finish generating non-atom graph')

        self.u_w_pooling_graph = to_tensor(uw_graph).to(device)
        self.i_w_pooling_graph = to_tensor(iw_graph).to(device)
        print('finish generating pooling graph')

        self.num_layers = num_layers

        self.dnns_atom = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])
        self.dnns_non_atom = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])

        # pretrain
        if pretrain is not None:
            self.users_feature.data = F.normalize(pretrain['users_feature'])
            self.items_feature.data = F.normalize(pretrain['items_feature'])
            self.words_feature.data = F.normalize(pretrain['words_feature'])

    def ww_propagate(self, graph, feature, dnns):
        all_features = [feature]
        for i in range(self.num_layers):
            feature = torch.cat([F.relu(dnns[i](torch.matmul(graph, feature))), feature], 1)
            all_features.append(F.normalize(feature))

        all_features = torch.cat(all_features, 1)
        return all_features

    def ui_propagate(self, graph, users_feature, items_feature, dnns):
        features = torch.cat((users_feature, items_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = torch.cat([F.relu(dnns[i](torch.matmul(graph, features))), features], 1)
            all_features.append(F.normalize(features))

        all_features = torch.cat(all_features, 1)
        users_feature, items_feature = torch.split(
            all_features, (self.num_users, self.num_items), 0)
        return users_feature, items_feature

    def propagate(self):
        #  ==============================  word level propagation  ==============================
        atom_words_feature = self.ww_propagate(
            self.atom_graph, self.words_feature, self.dnns_atom)
        atom_users_feature = F.normalize(torch.matmul(self.u_w_pooling_graph, atom_words_feature))
        atom_items_feature = F.normalize(torch.matmul(self.i_w_pooling_graph, atom_words_feature))

        #  ==============================  item level propagation  ==============================
        non_atom_users_feature, non_atom_items_feature = self.ui_propagate(
            self.non_atom_graph, self.users_feature, self.items_feature, self.dnns_non_atom)

        # users_feature = [atom_users_feature, non_atom_users_feature]
        # items_feature = [atom_items_feature, non_atom_items_feature]
        users_feature = torch.cat((atom_users_feature, non_atom_users_feature), 1)
        items_feature = torch.cat((atom_items_feature, non_atom_items_feature), 1)

        return users_feature, items_feature

    def predict(self, users_feature, items_feature):
        norm_users_feature = torch.sqrt(
            torch.sum(torch.square(users_feature), axis=1))
        norm_items_feature = torch.sqrt(
            torch.sum(torch.square(items_feature), axis=1))
        score = torch.sum(
            torch.mul(users_feature, items_feature), axis=1
        ) / (norm_users_feature * norm_items_feature)

        score = torch.maximum(torch.zeros_like(score) + 1e-6, score)
        return score

    def regularize(self, users_feature, items_feature):
        return super().regularize(users_feature, items_feature)

    def forward(self, u, i):
        users_feature, items_feature = self.propagate()
        users_feature = users_feature[u]
        items_feature = items_feature[i]
        pred = self.predict(users_feature, items_feature)
        regularization = self.regularize(users_feature, items_feature)
        return pred, regularization

        
