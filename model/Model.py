import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from model.BaseModel import BaseModel
from torch.nn import functional as F
from torch.utils.data.sampler import RandomSampler


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
    def __init__(self, embedding_size, data_size, link, raw_graph, num_layers, device, pretrain=None):
        super().__init__(embedding_size, data_size, create_embeddings=True)

        self.link = link
        self.device = device
        raw_graph_A, raw_graph_B = raw_graph
        assert isinstance(raw_graph_A, list)
        assert isinstance(raw_graph_B, list)
        self.num_layers = num_layers
        self.atom_graph_A, self.non_atom_graph_A, self.u_w_pooling_graph_A, self.i_w_pooling_graph_A = self.graphProcess(
            raw_graph_A, 'A', device)
        self.atom_graph_B, self.non_atom_graph_B, self.u_w_pooling_graph_B, self.i_w_pooling_graph_B = self.graphProcess(
            raw_graph_B, 'B', device)

        self.dnns_atom_A = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])
        self.dnns_non_atom_A = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])

        self.dnns_atom_B = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])
        self.dnns_non_atom_B = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])

        # pretrain
        if pretrain is not None:
            self.users_feature_A.data = F.normalize(pretrain['users_feature_A'])
            self.items_feature_A.data = F.normalize(pretrain['items_feature_A'])
            self.words_feature_A.data = F.normalize(pretrain['words_feature_A'])
            self.users_feature_B.data = F.normalize(pretrain['users_feature_B'])
            self.items_feature_B.data = F.normalize(pretrain['items_feature_B'])
            self.words_feature_B.data = F.normalize(pretrain['words_feature_B'])

        t = int((num_layers + 1) * (num_layers + 2) / 2)
        self.map = nn.Parameter(
            torch.randn(embedding_size * t, embedding_size * t),
            requires_grad=True
        )

        self.attn_A_A = nn.Parameter(
            torch.randn(embedding_size * t),
            requires_grad=True
        )

        self.attn_B_B = nn.Parameter(
            torch.randn(embedding_size * t),
            requires_grad=True
        )


    def graphProcess(self, raw_graph, domain, device):
        ui_graph, uw_graph, iw_graph, ww_graph = raw_graph

        if domain == 'A':
            num_users = self.num_users_A
            num_items = self.num_items_A
            num_words = self.num_words_A
        elif domain == 'B':
            num_users = self.num_users_B
            num_items = self.num_items_B
            num_words = self.num_words_B
        else:
            raise ValueError(r"non-exist domain")

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

        if ww_graph.shape == (num_words, num_words):
            atom_graph = ww_graph
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
        print('finish generating atom graph')

        if ui_graph.shape == (num_users, num_items) \
                and uu_graph.shape == (num_users, num_users) \
                    and ii_graph.shape == (num_items, num_items):
            # add self-loop
            non_atom_graph = sp.bmat([[uu_graph, ui_graph],
                                 [ui_graph.T, ii_graph]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        non_atom_graph = to_tensor(laplace_transform(non_atom_graph)).to(device)
        print('finish generating non-atom graph')

        u_w_pooling_graph = to_tensor(uw_graph).to(device)
        i_w_pooling_graph = to_tensor(iw_graph).to(device)
        print('finish generating pooling graph')

        return atom_graph, non_atom_graph, u_w_pooling_graph, i_w_pooling_graph

    def ww_propagate(self, graph, feature, dnns):
        all_features = [feature]
        for i in range(self.num_layers):
            feature = torch.cat([F.relu(dnns[i](torch.matmul(graph, feature))), feature], 1)
            all_features.append(F.normalize(feature))

        all_features = torch.cat(all_features, 1)
        return all_features

    def ui_propagate(self, graph, users_feature, items_feature, dnns, domain):
        if domain == 'A':
            num_users = self.num_users_A
            num_items = self.num_items_A
        elif domain == 'B':
            num_users = self.num_users_B
            num_items = self.num_items_B
        else:
            raise ValueError(r"non-exist domain")
        features = torch.cat((users_feature, items_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = torch.cat([F.relu(dnns[i](torch.matmul(graph, features))), features], 1)
            all_features.append(F.normalize(features))

        all_features = torch.cat(all_features, 1)
        users_feature, items_feature = torch.split(
            all_features, (num_users, num_items), 0)
        return users_feature, items_feature

    def transfer(self, feature, aug, domain):
        if domain == 'A':
            mapping = self.map
            attn = self.attn_A_A
        elif domain == 'B':
            mapping = torch.linalg.inv(self.map)
            attn = self.attn_B_B
        else:
            raise ValueError(r"non-exist domain")
        ind = torch.tensor(list(RandomSampler(feature, True, 2))).to(self.device)
        ind = ind.reshape(-1, 1)
        pro = torch.mm(feature, mapping)
        dis = torch.sqrt(torch.sum(torch.square(pro[ind] - aug), axis=2))
        match = torch.argmin(dis, dim=1)
        ind = ind.reshape(-1)
        feature[ind] = torch.add(
            feature[ind] * attn,
            aug[match] * (1-attn)
        )
        return feature

    def propagate(self):
        #  ==============================  word level propagation  ==============================
        atom_words_feature_A = self.ww_propagate(
            self.atom_graph_A, self.words_feature_A, self.dnns_atom_A)
        atom_words_feature_B = self.ww_propagate(
            self.atom_graph_B, self.words_feature_B, self.dnns_atom_B)
        ind = torch.tensor(list(RandomSampler(self.link, True, 1))).to(self.device)
        words_A, words_B = self.link[ind][:,0], self.link[ind][:,1]
        dis = torch.mm(atom_words_feature_A[words_A], self.map)
        dis = dis - atom_words_feature_B[words_B]
        dis = torch.square(dis)
        dis = torch.sum(dis, axis=1)
        dis = torch.sqrt(dis)

        atom_words_feature_A = self.transfer(atom_words_feature_A, atom_words_feature_B, 'A')
        atom_words_feature_B = self.transfer(atom_words_feature_B, atom_words_feature_A, 'B')
        
        atom_users_feature_A = F.normalize(torch.matmul(self.u_w_pooling_graph_A, atom_words_feature_A))
        atom_items_feature_A = F.normalize(torch.matmul(self.i_w_pooling_graph_A, atom_words_feature_A))
        atom_users_feature_B = F.normalize(torch.matmul(self.u_w_pooling_graph_B, atom_words_feature_B))
        atom_items_feature_B = F.normalize(torch.matmul(self.i_w_pooling_graph_B, atom_words_feature_B))

        #  ==============================  item level propagation  ==============================
        non_atom_users_feature_A, non_atom_items_feature_A = self.ui_propagate(
            self.non_atom_graph_A, self.users_feature_A, self.items_feature_A, self.dnns_non_atom_A, 'A')

        non_atom_users_feature_B, non_atom_items_feature_B = self.ui_propagate(
            self.non_atom_graph_B, self.users_feature_B, self.items_feature_B, self.dnns_non_atom_B, 'B')
            
        users_feature_A = torch.cat((atom_users_feature_A, non_atom_users_feature_A), 1)
        items_feature_A = torch.cat((atom_items_feature_A, non_atom_items_feature_A), 1)

        users_feature_B = torch.cat((atom_users_feature_B, non_atom_users_feature_B), 1)
        items_feature_B = torch.cat((atom_items_feature_B, non_atom_items_feature_B), 1)

        return users_feature_A, items_feature_A, users_feature_B, items_feature_B, dis

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

    def forward(self, u, i, domain):
        users_feature_A, items_feature_A, users_feature_B, items_feature_B, dis = self.propagate()
        if domain == 'A':
            users = users_feature_A
            items = items_feature_A
        elif domain == 'B':
            users = users_feature_B
            items = items_feature_B
        else:
            raise ValueError(r"non-exist domain")
        users = users[u]
        items = items[i]
        pred = self.predict(users, items)
        regularization = self.regularize(users, items)
        return pred, regularization, torch.mean(dis)

        
