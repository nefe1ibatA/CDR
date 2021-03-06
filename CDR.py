import torch
import torch.nn as nn
import numpy as np
import argparse
import heapq
import math
import scipy.io as scio
import matplotlib.pyplot as plt
from Dataset import Dataset
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from model.Model import Model
from sklearn.manifold import TSNE
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import preprocess_adj
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
allResults_A = []
allResults_B = []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-maxEpochs',
                        action='store',
                        dest='maxEpochs',
                        default=10)
    parser.add_argument('-lr',
                        action='store',
                        dest='lr',
                        default=0.0001)                   
    parser.add_argument('-lambdad',
                        action='store',
                        dest='lambdad',
                        default=0.001)
    parser.add_argument('-lambdad2',
                        action='store',
                        dest='lambdad2',
                        default=0.00005)
    parser.add_argument('-lambdad3',
                        action='store',
                        dest='lambdad3',
                        default=0.00005)
    parser.add_argument('-batchSize',
                        action='store',
                        dest='batchSize',
                        default=512)
    parser.add_argument('-negNum',
                        action='store',
                        dest='negNum',
                        default=7)
    parser.add_argument('-emb_dim',
                        action='store',
                        dest='emb_dim',
                        default=8)
    parser.add_argument('-topK',
                        action='store',
                        dest='topK',
                        default=10)
    args = parser.parse_args()

    classifier = trainer(args)

    classifier.run()


class trainer:
    def __init__(self, args):
        self.maxEpochs = args.maxEpochs
        self.lr = args.lr
        self.batchSize = args.batchSize
        self.lambdad = args.lambdad
        self.lambdad2 = args.lambdad2
        self.lambdad3 = args.lambdad3
        self.negNum = args.negNum
        self.emb_dim = args.emb_dim
        self.topK = args.topK

        with open('./data/clean/dataset_size_A.txt', 'r') as f:
            self.size_A = [int(s) for s in f.readline().split('\t')][:3]
        with open('./data/clean/dataset_size_B.txt', 'r') as f:
            self.size_B = [int(s) for s in f.readline().split('\t')][:3]
        self.link = []
        with open('./data/clean/common_words.txt', 'r') as f:
            for line in f.readlines():
                s = line.split('\t')
                self.link.append((int(s[0]), int(s[1])))
        self.link = torch.tensor(self.link)

        self.link_u = []
        with open('./data/clean/common_users.txt', 'r') as f:
            for line in f.readlines():
                s = line.split('\t')
                self.link_u.append((int(s[0]), int(s[1])))
        self.link_u = torch.tensor(self.link_u)

        # A
        self.dataset_A = Dataset('ui_graph_A', self.size_A)
        self.train_A, self.test_A = self.dataset_A.train, self.dataset_A.test
        self.testNeg_A = self.dataset_A.getTestNeg(self.test_A, 99)

        self.uw_dataset_A = Dataset('uw_graph_A', self.size_A)
        self.iw_dataset_A = Dataset('iw_graph_A', self.size_A)
        self.ww_dataset_A = Dataset('ww_graph_A', self.size_A)

        ui_graph_A = self.dataset_A.getgraph()
        uw_graph_A = self.uw_dataset_A.getgraph()
        iw_graph_A = self.iw_dataset_A.getgraph()
        ww_graph_A = self.ww_dataset_A.getgraph()
        self.raw_graph_A = [ui_graph_A, uw_graph_A, iw_graph_A, ww_graph_A]

        # B
        self.dataset_B = Dataset('ui_graph_B', self.size_B)
        self.train_B, self.test_B = self.dataset_B.train, self.dataset_B.test
        self.testNeg_B = self.dataset_B.getTestNeg(self.test_B, 99)

        self.uw_dataset_B = Dataset('uw_graph_B', self.size_B)
        self.iw_dataset_B = Dataset('iw_graph_B', self.size_B)
        self.ww_dataset_B = Dataset('ww_graph_B', self.size_B)

        ui_graph_B = self.dataset_B.getgraph()
        uw_graph_B = self.uw_dataset_B.getgraph()
        iw_graph_B = self.iw_dataset_B.getgraph()
        ww_graph_B = self.ww_dataset_B.getgraph()
        self.raw_graph_B = [ui_graph_B, uw_graph_B, iw_graph_B, ww_graph_B]

        self.size = [self.size_A, self.size_B]
        self.raw_graph = [self.raw_graph_A, self.raw_graph_B]


    def evaluate(self, model, testUser, testItem, domain, topK):
        HR = []
        NDCG = []

        print('testUser : ', len(testUser))
        for i in tqdm(range(len(testUser))):
            target = testItem[i][0]
            # pred, _, _ = model(testUser[i], testItem[i], domain)
            pred, _, _, _ = model(testUser[i], testItem[i], domain)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = pred[j]

            ranklist = heapq.nlargest(topK,
                                      item_score_dict,
                                      key=item_score_dict.get)

            tmp_HR = 0
            for item in ranklist:
                if item == target:
                    tmp_HR = 1

            HR.append(tmp_HR)

            tmp_NDCG = 0
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == target:
                    tmp_NDCG = math.log(2) / math.log(i + 2)

            NDCG.append(tmp_NDCG)

        HR = np.mean(HR)
        NDCG = np.mean(NDCG)

        return HR, NDCG

    def run(self):
        pretrainA = torch.load('pretrain/A')
        pretrainB = torch.load('pretrain/B')
        pretrain = {}
        pretrain['users_feature_A'] = pretrainA['users_feature_A']
        pretrain['items_feature_A'] = pretrainA['items_feature_A']
        pretrain['words_feature_A'] = pretrainA['words_feature_A']
        pretrain['users_feature_B'] = pretrainB['users_feature_B']
        pretrain['items_feature_B'] = pretrainB['items_feature_B']
        pretrain['words_feature_B'] = pretrainB['words_feature_B']
        model = Model(self.emb_dim, self.size, self.link, self.link_u, self.raw_graph, 2, device, pretrain)
        model = model.to(device)
        writer = SummaryWriter('runs/latest')
        print('training on device:', device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_HR_A = -1
        best_NDCG_A = -1
        best_epoch_A = -1

        best_HR_B = -1
        best_NDCG_B = -1
        best_epoch_B = -1 

        model.train()
        topK = self.topK
        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch + 1, "=" * 20)

            # A
            train_u_A, train_i_A, train_r_A = self.dataset_A.getInstances(
                self.train_A, self.negNum)
            train_len_A = len(train_u_A)
            shuffle_idx = np.random.permutation(np.arange(train_len_A))
            train_u_A = train_u_A[shuffle_idx]
            train_i_A = train_i_A[shuffle_idx]
            train_r_A = train_r_A[shuffle_idx]
            num_batches = train_len_A // self.batchSize + 1
            loss_A = torch.zeros(0).to(device)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len_A, (i + 1) * self.batchSize])

                if min_idx < train_len_A:
                    train_u_A_batch = torch.tensor(train_u_A[min_idx:max_idx]).to(device)
                    train_i_A_batch = torch.tensor(train_i_A[min_idx:max_idx]).to(device)
                    train_r_A_batch = torch.tensor(train_r_A[min_idx:max_idx]).to(device)

                    # pred, regularizer, dis_t = model(train_u_A_batch, train_i_A_batch, 'A')
                    pred, regularizer, dis_t, dis_u = model(train_u_A_batch, train_i_A_batch, 'A')

                    regRate = train_r_A_batch / 5.0
                    # batchloss_A = torch.sum(-(regRate * torch.log(
                        # pred) + (1 - regRate) * torch.log(
                        # 1 - pred))) + self.lambdad * regularizer + self.lambdad2 * dis_t
                    batchloss_A = torch.sum(-(regRate * torch.log(
                        pred) + (1 - regRate) * torch.log(
                        1 - pred))) + self.lambdad * regularizer + self.lambdad2 * dis_t + self.lambdad3 * dis_u

                    optimizer.zero_grad()
                    batchloss_A.backward()
                    optimizer.step()
                    loss_A = torch.cat((loss_A, torch.tensor([batchloss_A]).to(device)))
            loss_A = torch.mean(loss_A)
            print("Mean loss A in epoch {} is: {}\n".format(epoch + 1, loss_A))
            writer.add_scalar('loss/loss_A', loss_A, global_step=epoch)

            # B
            train_u_B, train_i_B, train_r_B = self.dataset_B.getInstances(
                self.train_B, self.negNum)
            train_len_B = len(train_u_B)
            shuffle_idx = np.random.permutation(np.arange(train_len_B))
            train_u_B = train_u_B[shuffle_idx]
            train_i_B = train_i_B[shuffle_idx]
            train_r_B = train_r_B[shuffle_idx]
            num_batches = train_len_B // self.batchSize + 1
            loss_B = torch.zeros(0).to(device)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len_B, (i + 1) * self.batchSize])

                if min_idx < train_len_B:
                    train_u_B_batch = torch.tensor(train_u_B[min_idx:max_idx]).to(device)
                    train_i_B_batch = torch.tensor(train_i_B[min_idx:max_idx]).to(device)
                    train_r_B_batch = torch.tensor(train_r_B[min_idx:max_idx]).to(device)

                    # pred, regularizer, dis_t = model(train_u_B_batch, train_i_B_batch, 'B')
                    pred, regularizer, dis_t, dis_u = model(train_u_B_batch, train_i_B_batch, 'B')

                    regRate = train_r_B_batch / 5.0
                    # batchloss_B = torch.sum(-(regRate * torch.log(
                        # pred) + (1 - regRate) * torch.log(
                        # 1 - pred))) + self.lambdad * regularizer + self.lambdad2 * dis_t
                    batchloss_B = torch.sum(-(regRate * torch.log(
                        pred) + (1 - regRate) * torch.log(
                        1 - pred))) + self.lambdad * regularizer + self.lambdad2 * dis_t + self.lambdad3 * dis_u

                    optimizer.zero_grad()
                    batchloss_B.backward()
                    optimizer.step()
                    loss_B = torch.cat((loss_B, torch.tensor([batchloss_B]).to(device)))
            loss_B = torch.mean(loss_B)
            print("Mean loss B in epoch {} is: {}\n".format(epoch + 1, loss_B))
            writer.add_scalar('loss/loss_B', loss_B, global_step=epoch)

            model.eval()

            testUser = torch.tensor(self.testNeg_A[0]).to(device)
            testItem = torch.tensor(self.testNeg_A[1]).to(device)
            HR_A, NDCG_A = self.evaluate(model, testUser, testItem, 'A', topK)
            writer.add_scalar('others/HR_A', HR_A, global_step=epoch)
            writer.add_scalar('others/NDCG_A', NDCG_A, global_step=epoch)

            allResults_A.append([epoch + 1, topK, HR_A, NDCG_A, loss_A.detach().cpu().numpy()])
            print(
                "Domain A Epoch: ", epoch + 1,
                "TopK: {} HR: {}, NDCG: {}".format(
                    topK, HR_A, NDCG_A))
            if HR_A > best_HR_A:
                best_HR_A = HR_A
                best_epoch_A = epoch + 1
            if NDCG_A > best_NDCG_A:
                best_NDCG_A = NDCG_A

            testUser = torch.tensor(self.testNeg_B[0]).to(device)
            testItem = torch.tensor(self.testNeg_B[1]).to(device)
            HR_B, NDCG_B = self.evaluate(model, testUser, testItem, 'B', topK)
            writer.add_scalar('others/HR_B', HR_B, global_step=epoch)
            writer.add_scalar('others/NDCG_B', NDCG_B, global_step=epoch)

            allResults_B.append([epoch + 1, topK, HR_B, NDCG_B, loss_B.detach().cpu().numpy()])
            print(
                "Domain B Epoch: ", epoch + 1,
                "TopK: {} HR: {}, NDCG: {}".format(
                    topK, HR_B, NDCG_B))
            if HR_B > best_HR_B:
                best_HR_B = HR_B
                best_epoch_B = epoch + 1
            if NDCG_B > best_NDCG_B:
                best_NDCG_B = NDCG_B

            with torch.no_grad():
                # userA, _, userB, _, _ = model.propagate()
                userA, _, userB, _, _, _ = model.propagate()
                userA = [i.detach().cpu().numpy().reshape(-1) for i in userA]
                userB = [i.detach().cpu().numpy().reshape(-1) for i in userB]

                tsne = TSNE(n_components=2, init='pca', random_state=0)
                userA = tsne.fit_transform(userA)
                userB = tsne.fit_transform(userB)

                fig0 = plt.figure()
                t1 = plt.scatter(userA[:, 0], userA[:, 1], marker='o', c='r', s=10)  # marker:????????? c:????????? s:?????????
                t2 = plt.scatter(userB[:, 0], userB[:, 1], marker='o', c='b', s=10)
                # plt.legend((t1, t2), ('userA', 'userB'))
                plt.savefig('tsne/epoch {}'.format(epoch+1))
        print(
            "Domain A: Best HR: {}, NDCG: {} At Epoch {}".format(best_HR_A, best_NDCG_A, best_epoch_A))
        print(
            "Domain B: Best HR: {}, NDCG: {} At Epoch {}".format(best_HR_B, best_NDCG_B, best_epoch_B))

        bestPerformance = [[best_HR_A, best_NDCG_A, best_epoch_A],
                           [best_HR_A, best_NDCG_A, best_epoch_A]]

        matname = 'result.mat'
        scio.savemat(
            matname, {
            'allResults_A': allResults_A,
            'allResults_B': allResults_B,
            'bestPerformance': bestPerformance
            }
        )
        print("Training complete!")


if __name__ == '__main__':
    main()
