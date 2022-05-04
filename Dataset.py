import numpy as np
import os
import scipy.sparse as sp

class Dataset:
	def __init__(self, path, size):
		self.path = path
		self.data = self.getData()
		self.size = size
		if 'ui_graph' in self.path:
			self.shape = (size[0], size[1])
			self.train, self.test = self.getTrainTest()
			self.trainDict = self.getTrainDict()


	def getData(self):
		data = []
		with open(os.path.join('./data/clean/', '{}.txt'.format(self.path)), 'r') as f:
			for line in f.readlines():
				s = line.split('\t')
				data.append([int(s[0]), int(s[1]), float(s[2])])
		return data	

	def getgraph(self):
		self.pairs = [i[0:2] for i in self.data]
		self.weights = [i[2] for i in self.data]
		indice = np.array(self.pairs, dtype=np.int32)
		values = np.array(self.weights, dtype=np.float32)
		if 'ui_graph' in self.path:
			graph = sp.coo_matrix(
            	(values, (indice[:, 0], indice[:, 1])), shape=(self.size[0], self.size[1])).tocsr()
		elif 'uw_graph' in self.path:
			graph = sp.coo_matrix(
            	(values, (indice[:, 0], indice[:, 1])), shape=(self.size[0], self.size[2])).tocsr()
		elif 'iw_graph' in self.path:
			graph = sp.coo_matrix(
            	(values, (indice[:, 0], indice[:, 1])), shape=(self.size[1], self.size[2])).tocsr()
		elif 'ww_graph' in self.path:
			graph = sp.coo_matrix(
            	(values, (indice[:, 0], indice[:, 1])), shape=(self.size[2], self.size[2])).tocsr()
		else:
			raise ValueError(r"non-exist graph type")
		return graph

	def getTrainTest(self):
		data = self.data
		data = sorted(data, key=lambda x: x[0])
		train = []
		test = []
		for i in range(len(data)-1):
			user = data[i][0]
			item = data[i][1]
			rate = data[i][2]
			if data[i][0] != data[i+1][0]:
				test.append((user, item, rate))
			else:
				train.append((user, item, rate))

		test.append((data[-1][0]-1, data[-1][1]-1, data[1][2]))
		return train, test

	def getTrainDict(self):
		dataDict = {}
		for i in self.train:
			dataDict[(i[0], i[1])] = i[2]
		return dataDict

	def getInstances(self, data, negNum):
		user = []
		item = []
		rate = []
		for i in data:
			user.append(i[0])
			item.append(i[1])
			rate.append(i[2])
			for t in range(negNum):
				j = np.random.randint(self.shape[1])
				while (i[0], j) in self.trainDict:
					j = np.random.randint(self.shape[1])
				user.append(i[0])
				item.append(j)
				rate.append(0.0)
		return np.array(user), np.array(item), np.array(rate)

	def getTestNeg(self, testData, negNum):
		user = []
		item = []
		for s in testData:
			tmp_user = []
			tmp_item = []
			u = s[0]
			i = s[1]
			tmp_user.append(u)
			tmp_item.append(i)
			neglist = set()
			neglist.add(i)
			for t in range(negNum):
				j = np.random.randint(self.shape[1])
				while (u, j) in self.trainDict or j in neglist:
					j = np.random.randint(self.shape[1])
				neglist.add(j)
				tmp_user.append(u)
				tmp_item.append(j)
			user.append(tmp_user)
			item.append(tmp_item)
		return [np.array(user), np.array(item)]
