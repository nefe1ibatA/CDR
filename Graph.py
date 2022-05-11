from preprocess.datareader import Datareader
from preprocess.RatingGraph import RatingGraphBuilder
from preprocess.ReviewGraph import ReviewGraphBuilder
import os
from typing import List

def g2f(graph, name):
    filePath = './data/clean/' + name + '.txt'
    f = open(filePath, 'w+')
    for (u, v, wt) in graph:
        s = str(u) + '\t' + str(v) + '\t' + str(wt) + '\n'
        f.write(s)
    f.close()

def graphGen(user_rating_dict, user_review_dict, item_user_dict, domain):
    ui_graph = []
    uw_graph = []
    iw_graph = []
    ww_graph = []
    dataset = RatingGraphBuilder(
            user_rating_dict, user_review_dict, item_user_dict)
    ReviewGraph = ReviewGraphBuilder(user_review_dict, item_user_dict)
    ReviewGraph = ReviewGraph.adj()
    num_users = dataset.shape[0]
    num_docs = dataset.shape[0] + dataset.shape[1]
    num_words = ReviewGraph.number_of_nodes() - num_docs
    f = open(os.path.join('./data/clean/dataset_size_' + domain + '.txt'), 'w+')
    s = str(num_users) + '\t' + str(num_docs - num_users) + '\t' + str(num_words) + '\n'
    f.write(s)
    f.close()
    for (u, v, wt) in ReviewGraph.edges.data('weight'):
        if u < num_users:
            uw_graph.append((u, v - num_docs, wt))
        elif u >= num_users and u < num_docs:
            iw_graph.append((u - num_users, v - num_docs, wt))
        else:
            ww_graph.append((u - num_docs, v - num_docs, wt))
    RatingGraph, _ = dataset.getData(user_rating_dict, user_review_dict, item_user_dict)
    for (u, v, wt) in RatingGraph:
        ui_graph.append((u, v - num_users, wt))
    g2f(ui_graph, 'ui_graph_' + domain)
    g2f(uw_graph, 'uw_graph_' + domain)
    g2f(iw_graph, 'iw_graph_' + domain)
    g2f(ww_graph, 'ww_graph_' + domain)

def main(dataName_A, dataName_B):
    dr = Datareader(dataName_A, dataName_B)
    A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
        B_user_rating_dict, B_user_review_dict, B_item_user_dict \
            = dr.read_data()

    graphGen(A_user_rating_dict, A_user_review_dict, A_item_user_dict, 'A')
    graphGen(B_user_rating_dict, B_user_review_dict, B_item_user_dict, 'B')
    calCommon(A_user_review_dict, A_item_user_dict, B_user_review_dict, B_item_user_dict)


class CommonWords:
    def __init__(self, w: str, A: int, B: int) -> None:
        self.word: str = w
        self.Aidx: int = A
        self.Bidx: int = B

def calCommon(A_user_review_dict, A_item_user_dict, B_user_review_dict, B_item_user_dict):

    A_ReviewGraph = ReviewGraphBuilder(A_user_review_dict, A_item_user_dict)
    A_w2i: dict = A_ReviewGraph.getWordsidx()

    B_ReviewGraph = ReviewGraphBuilder(B_user_review_dict, B_item_user_dict)
    B_w2i: dict = B_ReviewGraph.getWordsidx()
    commonList: List[CommonWords] = []
    for A_w, A_i in A_w2i.items():
        if A_w in B_w2i:
            commonList.append(CommonWords(A_w, A_i, B_w2i[A_w]))
    with open('./data/clean/common_words.txt', 'w+') as t:
        t.write()
    return commonList



if __name__ == '__main__':
    main('Appliances', 'Movies_and_TV')
    t = calCommon('Appliances', 'Movies_and_TV')
