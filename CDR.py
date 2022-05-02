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
allResults = []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main(dataName_A, dataName_B):
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-dataName_A',
                        action='store',
                        dest='dataName_A',
                        default=dataName_A)
    parser.add_argument('-dataName_B',
                        action='store',
                        dest='dataName_B',
                        default=dataName_B)
    parser.add_argument('-maxEpochs',
                        action='store',
                        dest='maxEpochs',
                        default=100)
    parser.add_argument('-lr',
                        action='store',
                        dest='lr',
                        default=0.001)
    parser.add_argument('-lr_G',
                        action='store',
                        dest='lr_G',
                        default=0.001)
    parser.add_argument('-lr_D',
                        action='store',
                        dest='lr_D',
                        default=0.0000)                    
    parser.add_argument('-lambdad',
                        action='store',
                        dest='lambdad',
                        default=0.001)
    parser.add_argument('-batchSize',
                        action='store',
                        dest='batchSize',
                        default=4096)
    parser.add_argument('-negNum',
                        action='store',
                        dest='negNum',
                        default=7)
    parser.add_argument('-emb_dim',
                        action='store',
                        dest='emb_dim',
                        default=200)
    parser.add_argument('-topK',
                        action='store',
                        dest='topK',
                        default=10)
    args = parser.parse_args()

    classifier = trainer(args)

    classifier.run()


class trainer:
    def __init__(self, args):
        self.dataName_A = args.dataName_A
        self.dataName_B = args.dataName_B
        self.maxEpochs = args.maxEpochs
        self.lr = args.lr
        self.batchSize = args.batchSize
        self.lambdad = args.lambdad
        self.negNum = args.negNum
        self.emb_dim = args.emb_dim
        self.topK = args.topK

        with open('./data/clean/dataset_size.txt', 'r') as f:
            self.size = [int(s) for s in f.readline().split('\t')][:3]

        self.dataset = Dataset('ui_graph', self.size)
        self.train, self.test = self.dataset.train, self.dataset.test
        self.testNeg = self.dataset.getTestNeg(self.test, 99)

        self.uw_dataset = Dataset('uw_graph', self.size)
        self.iw_dataset = Dataset('iw_graph', self.size)
        self.ww_dataset = Dataset('ww_graph', self.size)

        ui_graph = self.dataset.getgraph()
        uw_graph = self.uw_dataset.getgraph()
        iw_graph = self.iw_dataset.getgraph()
        ww_graph = self.ww_dataset.getgraph()
        self.raw_graph = [ui_graph, uw_graph, iw_graph, ww_graph]


    def run(self):
        model = Model(self.emb_dim, self.size, self.raw_graph, 2, device)
        model = model.to(device)
        writer = SummaryWriter('runs/latest')
        print('training on device:', device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_HR = -1
        best_NDCG = -1
        best_epoch = -1

        model.train()
        topK = self.topK
        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch + 1, "=" * 20)
            train_u, train_i, train_r = self.dataset.getInstances(
                self.train, self.negNum)
            train_len = len(train_u)
            shuffle_idx = np.random.permutation(np.arange(train_len))
            train_u = train_u[shuffle_idx]
            train_i = train_i[shuffle_idx]
            train_r = train_r[shuffle_idx]

            num_batches = train_len // self.batchSize + 1

            loss = torch.zeros(0).to(device)
            for i in range(num_batches):
                min_idx = i * self.batchSize
                max_idx = np.min([train_len, (i + 1) * self.batchSize])

                if min_idx < train_len:
                    train_u_batch = torch.tensor(train_u[min_idx:max_idx]).to(device)
                    train_i_batch = torch.tensor(train_i[min_idx:max_idx]).to(device)
                    train_r_batch = torch.tensor(train_r[min_idx:max_idx]).to(device)

                    pred, regularizer = model(train_u_batch, train_i_batch)

                    regRate = train_r_batch / 5.0
                    batchloss = torch.sum(-(regRate * torch.log(
                        pred) + (1 - regRate) * torch.log(
                        1 - pred))) + self.lambdad * regularizer

                    optimizer.zero_grad()
                    batchloss.backward()
                    optimizer.step()
                    loss = torch.cat((loss, torch.tensor([batchloss]).to(device)))

            loss = torch.mean(loss)
            print("Mean loss in epoch {} is: {}\n".format(epoch + 1, loss))
            writer.add_scalar('loss/loss', loss, global_step=epoch)

            model.eval()

            HR = []
            NDCG = []
            testUser = self.testNeg[0]
            testItem = self.testNeg[1]
            print('testUser : ', len(testUser))
            for i in tqdm(range(len(testUser))):
                target = testItem[i][0]
                pred, _ = model(testUser[i], testItem[i])

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

            writer.add_scalar('others/HR', HR, global_step=epoch)
            writer.add_scalar('others/NDCG', NDCG, global_step=epoch)

            allResults.append([epoch + 1, topK, HR, NDCG, loss.detach().cpu().numpy()])
            print(
                "Domain A Epoch: ", epoch + 1,
                "TopK: {} HR: {}, NDCG: {}".format(
                    topK, HR, NDCG))


            if HR > best_HR:
                best_HR = HR
                best_epoch = epoch + 1
            if NDCG > best_NDCG:
                best_NDCG = NDCG
                best_epoch = epoch + 1


        print(
            "Domain A: Best HR: {}, NDCG: {} At Epoch {}".format(best_HR, best_NDCG, best_epoch))

        bestPerformance = [[best_HR, best_NDCG, best_epoch]]

        model.eval()
        with torch.no_grad():
            userVecs = [model(u, testItem[0], 'A')[0] for u in testUser]
            itemVecs = [model(testUser[0], i, 'A')[1] for i in testItem]
            userVecs = [i.detach().cpu().numpy().reshape(-1) for i in userVecs]
            itemVecs = [i.detach().cpu().numpy().reshape(-1) for i in itemVecs]

            tsne = TSNE(n_components=2, init='pca', random_state=0)
            user2D = tsne.fit_transform(userVecs)
            item2D = tsne.fit_transform(itemVecs)

            fig0 = plt.figure()
            t1 = plt.scatter(user2D[:, 0], user2D[:, 1], marker='x', c='r', s=20)  # marker:点符号 c:点颜色 s:点大小
            t2 = plt.scatter(item2D[:, 0], item2D[:, 1], marker='o', c='b', s=20)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend((t1, t2), ('user', 'item'))
            plt.show()

        matname = 'baseline_result.mat'
        scio.savemat(matname, {
            'allResults': allResults,
            'bestPerformance': bestPerformance
        })


if __name__ == '__main__':
    main('Appliances', 'Movies_and_TV')
    # main('Movies_and_TV', 'Appliances')
    # main('Arts_Crafts_and_Sewing_5', 'Luxury_Beauty_5')
    allResults = np.array(allResults)
    x_label = allResults[:, 0]
    y_topK = allResults[:, 1]
    y_HR = allResults[:, 2]
    y_NDCG = allResults[:, 3]
    y_loss = allResults[:, 4]

    fig1 = plt.figure()
    f1 = fig1.add_subplot(1, 1, 1)
    f1.set_title('Loss')
    f1.set_xlabel('Epoch')
    f1.set_ylabel('Loss')
    f1.grid()
    f1.plot(x_label, y_loss)
    fig1.savefig('Loss.jpg')

    fig2 = plt.figure()
    f2 = fig2.add_subplot(1, 1, 1)
    f2.set_title('HR')
    f2.set_xlabel('Epoch')
    f2.set_ylabel('HR')
    f2.grid()
    f2.plot(x_label, y_HR)
    fig2.savefig('HR.jpg')

    fig3 = plt.figure()
    f3 = fig3.add_subplot(1, 1, 1)
    f2.set_title('NDCG')
    f3.set_xlabel('Epoch')
    f3.set_ylabel('NDCG')
    f3.grid()
    f3.plot(x_label, y_NDCG)
    fig3.savefig('NDCG.jpg')

    plt.show()

# plt.plot()