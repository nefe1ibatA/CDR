from preprocess.datareader import Datareader
from preprocess.RatingGraph import RatingGraphBuilder
from preprocess.ReviewGraph import ReviewGraphBuilder
import os

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



if __name__ == '__main__':
    main('Appliances', 'Movies_and_TV')