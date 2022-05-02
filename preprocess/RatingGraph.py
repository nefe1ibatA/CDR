import networkx as nx
from utils import print_graph_detail

class RatingGraphBuilder(object):
	def __init__(self, user_rating_dict, user_review_dict, item_user_dict):
		self.graph = nx.Graph()
		self.data, self.adj = self.getData(
			user_rating_dict, user_review_dict, item_user_dict)

	def getData(self, user_rating_dict, user_review_dict, item_user_dict):
		maxr = 0.0
		data = []

		nodeset = [_key for _key in user_review_dict.keys()]
		nodeset += [_key for _key in item_user_dict.keys()]

		nodedict = {}

		for index, node in enumerate(nodeset):
			nodedict[node] = index

		u = 0
		for user, reviews in user_review_dict.items():
			u += 1
			str = ""
			for each in reviews:
				str += " "
				str += each[1]

		i = 0
		for item, reviews in item_user_dict.items():
			i += 1
			str = ""
			for each in reviews:
				str += " "
				str += each[1]

		self.shape = [u, i]
		print("user: {} item: {}".format(u, i))
		self.graph.add_nodes_from(range(u+i))

		for user, rating in user_rating_dict.items():
			for each in rating:
				data.append((nodedict[user], nodedict[each[0]], each[1]))
				if each[1] > maxr:
					maxr = each[1]

		self.maxRate = maxr

		data = sorted(data, key=lambda x: x[0])
		for i in range(len(data)-1):
			user = data[i][0]
			item = data[i][1]
			rate = data[i][2]
			if data[i][0] != data[i+1][0]:
				pass
			else:
				self.graph.add_weighted_edges_from([(user, item, rate / maxr)])

		print_graph_detail(self.graph)

		return data, self.graph

