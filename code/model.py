import torch
from dataloader import RecsysData
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from utils import cust_mul

class LightGCN(nn.Module):
    def __init__(self,  
                 dataset: RecsysData,
                 latent_dim=200, 
                 n_layers=3, 
                 keep_prob=0.8,
                 dropout_bool=False,
                 l2_w=1e-4):
        super(LightGCN, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout_bool = dropout_bool
        self.keep_prob = keep_prob
        self.Graph = dataset.getSparseGraph()
        self.num_users  = dataset.n_user
        self.num_items  = dataset.m_item
        self.l2_w = l2_w
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    

    def computer(self):    
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        if self.dropout_bool and self.training:
            g_droped = self.__dropout(self.keep_prob)     
        else:
            g_droped = self.Graph
        
        all_emb_list = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            all_emb_list.append(all_emb)

        all_emb = torch.mean(torch.stack(all_emb_list),0)
        #all_emb = all_emb_list[-1]
        users, items = torch.split(all_emb, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        return loss + self.l2_w * reg_loss
       
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class IMP_GCN(nn.Module):
    def __init__(self,  
                 dataset: RecsysData,
                 latent_dim=200, 
                 n_layers=6, 
                 keep_prob=0.9,
                 groups=4,
                 device=torch.device('cuda'),
                 dropout_bool=False,
                 l2_w=1e-4,
                 single=False):
        super(IMP_GCN, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout_bool = dropout_bool
        self.keep_prob = keep_prob
        self.Graph = dataset.getSparseGraph()
        self.num_users  = dataset.n_user
        self.num_items  = dataset.m_item
        self.groups = groups
        self.device = device
        self.l2_w = l2_w
        self.single = single
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc_g = torch.nn.Linear(self.latent_dim, self.groups)
        self.f = nn.Sigmoid()

        #nn.init.normal_(self.embedding_user.weight, std=0.1)
        #nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        #nn.init.xavier_uniform_(self.fc.weight, gain=1)
        #nn.init.xavier_uniform_(self.fc_g.weight, gain=1)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    

    def computer(self):    
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        if self.dropout_bool and self.training:
            g_droped = self.__dropout(self.keep_prob)     
        else:
            g_droped = self.Graph
        
        # Compute ego + side embeddings
        ego_embed = all_emb
        side_embed = torch.sparse.mm(g_droped, all_emb)
        
        temp = self.dropout(self.leaky(self.fc(ego_embed + side_embed)))
        group_scores = self.dropout(self.fc_g(temp))
        #group_scores = self.fc_g(temp)

        a_top, a_top_idx = torch.topk(group_scores, k=1, sorted=False)
        one_hot_emb = torch.eq(group_scores,a_top).float()

        u_one_hot, i_one_hot = torch.split(one_hot_emb, [self.num_users, self.num_items])
        i_one_hot = torch.ones(i_one_hot.shape).to(self.device)
        one_hot_emb = torch.cat([u_one_hot, i_one_hot]).t()

        # Create Subgraphs
        subgraph_list = []
        for g in range(self.groups):
            temp = cust_mul(g_droped, one_hot_emb[g], 1)
            temp = cust_mul(temp, one_hot_emb[g], 0)
            subgraph_list.append(temp)

        all_emb_list = [[None for _ in range(self.groups)] for _ in range(self.n_layers)]
        for g in range(0,self.groups):
            all_emb_list[0][g] = ego_embed
        
        for k in range(1,self.n_layers):
            for g in range(self.groups):
                all_emb_list[k][g] = torch.sparse.mm(subgraph_list[g], all_emb_list[k-1][g])

        
        all_emb_list = [torch.sum(torch.stack(x),0) for x in all_emb_list]
        
        if self.single:
            all_emb = all_emb_list[-1]
        else:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            all_emb_list = [x * w for x,w in zip(all_emb_list,weights)]
            all_emb = torch.sum(torch.stack(all_emb_list),0)
            #all_emb = torch.mean(torch.stack(all_emb_list),0)
            #all_emb = all_emb_list[-1]

        users, items = torch.split(all_emb, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        return loss + self.l2_w * reg_loss
       
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

