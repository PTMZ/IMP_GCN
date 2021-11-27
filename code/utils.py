import numpy as np
import torch
from dataloader import RecsysData

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def UniformSample(dataset:RecsysData):
    users = np.random.randint(0, dataset.n_user, dataset.trainSize)
    allPos = dataset.allPos
    S = []
    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        positem = posForUser[np.random.randint(0, len(posForUser))]
        negitem = np.random.randint(0, dataset.m_item)
        while negitem in posForUser:
            negitem = np.random.randint(0, dataset.m_item)

        S.append([user, positem, negitem])

    return np.array(S)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def generate_batches(tensors, batch_size):
    for i in range(0, len(tensors), batch_size):
        yield tensors[i:i + batch_size]

def minibatch(tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def cust_mul(s, d, dim):
    i = s._indices()
    v = s._values()
    dv = d[i[dim,:]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def calc_recall(ratings, test_data, k, users):
    num = ratings.sum(1)
    den = np.array([len(test_data[u]) for u in users]).astype('float')
    num[den == 0.] = 0.
    den[den == 0.] = 1.
    recall = np.sum(num/den)
    return recall

def calc_ndcg(ratings, test_data, k, users):
    test_matrix = np.zeros((len(users), k))
    for i,user in enumerate(users):
        length = k if k <= len(test_data[user]) else len(test_data[user])
        test_matrix[i, :length] = 1

    idcg = np.sum(test_matrix * 1./np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.] = 1.
    dcg = ratings*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = np.sum(dcg/idcg)
    return ndcg

def calc_ncrr(ratings, test_data, k, users):
    fractions = [1.0/n for n in range(1,k+1)]
    fractions = np.array(fractions)
    crr = ratings.dot(fractions.T)

    accum = np.cumsum(fractions)
    icrr = np.array([accum[min(len(test_data[u])-1, k-1)] for u in users])
    icrr[icrr == 0.] = 1.

    ncrr = np.sum(crr/icrr)
    return ncrr

def calc_hm(a, b, c):
    return 3/(1/a + 1/b + 1/c)

