from preprocess.datareader import Datareader
from preprocess.RatingGraph import RatingGraphBuilder
from preprocess.ReviewGraph import ReviewGraphBuilder

def g2f(graph, name):
    filePath = './data/clean/' + name + '.txt'
    f = open(filePath, 'w+')
    for (u, v, wt) in graph:
        s = str(u) + '\t' + str(v) + '\t' + str(wt) + '\n'
        f.write(s)
    f.close()

def main(dataName_A, dataName_B):
    dr = Datareader(dataName_A, dataName_B)
    A_user_rating_dict, A_user_review_dict, A_item_user_dict, \
        B_user_rating_dict, B_user_review_dict, B_item_user_dict \
            = dr.read_data()

    ui_graph = []
    uw_graph = []
    iw_graph = []
    ww_graph = []
    dataset_A = RatingGraphBuilder(
            A_user_rating_dict, A_user_review_dict, A_item_user_dict)
    ReviewGraph_A = ReviewGraphBuilder(A_user_review_dict, A_item_user_dict)
    ReviewGraph_A = ReviewGraph_A.adj()
    num_users = dataset_A.shape[0]
    num_docs = dataset_A.shape[0] + dataset_A.shape[1]
    num_words = ReviewGraph_A.number_of_nodes() - num_docs
    f = open('./data/clean/dataset_size.txt', 'w+')
    s = str(num_users) + '\t' + str(num_docs - num_users) + '\t' + str(num_words) + '\n'
    f.write(s)
    f.close()
    for (u, v, wt) in ReviewGraph_A.edges.data('weight'):
        if u < num_users:
            uw_graph.append((u, v - num_docs, wt))
        elif u >= num_users and u < num_docs:
            iw_graph.append((u - num_users, v - num_docs, wt))
        else:
            ww_graph.append((u - num_docs, v - num_docs, wt))
    RatingGraph_A, _ = dataset_A.getData(A_user_rating_dict, A_user_review_dict, A_item_user_dict)
    for (u, v, wt) in RatingGraph_A:
        ui_graph.append((u, v - num_users, wt))
    g2f(ui_graph, 'ui_graph')
    g2f(uw_graph, 'uw_graph')
    g2f(iw_graph, 'iw_graph')
    g2f(ww_graph, 'ww_graph')



if __name__ == '__main__':
    main('Appliances', 'Movies_and_TV')