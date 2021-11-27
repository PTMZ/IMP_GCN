
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from collections import defaultdict
from time import time

ALL_TRAIN = False

class RecsysData(Dataset):

    def __init__(self, path="../data/recsys2"):
        # train or test
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        print(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        self.trainSize = 0
        self.testSize = 0

        df = pd.read_csv(train_file)
        self.trainUniqueUsers = pd.unique(df["user_id"])
        self.trainUser = df["user_id"].to_numpy() - 1
        self.trainItem = df["item_id"].to_numpy() - 1
        self.trainSize = len(df)

        self.m_item = len(pd.unique(df["item_id"]))
        self.n_user = len(pd.unique(df["user_id"]))

        df = pd.read_csv(test_file)
        self.testUniqueUsers = pd.unique(df["user_id"])
        self.testUser = df["user_id"].to_numpy() - 1
        self.testItem = df["item_id"].to_numpy() - 1

        self.testSize = len(df)

        if ALL_TRAIN:
            self.trainUser = np.concatenate((self.trainUser, self.testUser), axis=0)
            self.trainItem = np.concatenate((self.trainItem, self.testItem), axis=0)
            self.trainSize = self.trainSize + self.testSize
        
        self.Graph = None
        print(f"{self.trainSize} interactions for training")
        print(f"{self.testSize} interactions for testing")
        print(f"Sparsity : {(self.trainSize + self.testSize) / self.n_user / self.m_item}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), 
                                        (self.trainUser, self.trainItem)),
                                        shape=(self.n_user, self.m_item))
        

        # pre-calculate
        self.allPos = self.getUserPosItems(list(range(self.n_user)))
        self.testDict = self.__build_test()
        print(f"Ready to go")

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def saveRatingMatrix(self):
        test_ratings = csr_matrix((np.ones(len(self.testUser)), 
                                        (self.testUser, self.testItem)),
                                        shape=(self.n_user, self.m_item))
        sp.save_npz(self.path + '/train_mat.npz', self.UserItemNet)
        sp.save_npz(self.path + '/test_mat.npz', test_ratings)


    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

        return self.Graph


    def __build_test(self):
        test_data = defaultdict(set)
        for user, item in zip(self.testUser, self.testItem):
            test_data[user].add(item)

        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

if __name__ == '__main__':
    d = RecsysData()
    g = d.getSparseGraph()