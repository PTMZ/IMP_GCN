import utils
SEED = 2021
utils.set_seed(SEED)

from dataloader import RecsysData, ALL_TRAIN
from model_configs import dataset, device, args1, args2
from model import LightGCN, IMP_GCN
import torch
import numpy as np
import time
from tqdm import tqdm


EPOCHS = 30
TEST_INTERVAL = 5
BATCH_SIZE = 1024

TOP_K = 50

def run_train(dataset, model, optimiser):
    model.train()
    num_batches = dataset.trainSize // BATCH_SIZE + 1
    mean_batch_loss = 0

    S = utils.UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long().to(device)
    posItems = torch.Tensor(S[:, 1]).long().to(device)
    negItems = torch.Tensor(S[:, 2]).long().to(device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    for i,(b_users, b_pos, b_neg) in enumerate(utils.minibatch((users,posItems,negItems), BATCH_SIZE)):
        #bpr_loss, reg_loss = model.bpr_loss(b_users, b_pos, b_neg)
        #loss = bpr_loss + reg_loss*L2_W
        loss = model.bpr_loss(b_users, b_pos, b_neg)
        mean_batch_loss += loss.cpu().item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"  Batch {i+1}/{num_batches}: loss = {loss.cpu().item():.8f}" + " " * 20, end='\r')
    

    return f"Final train loss: {mean_batch_loss/num_batches:.7f}"

def run_test(dataset, model):
    model.eval()
    print("-"*40)
    print("TEST RESULTS")
    print("-"*40)
    
    test_data = dataset.testDict
    test_users = list(test_data.keys())
    test_batch_size = 100
    cur_idx = 0
    
    t_recall = 0
    t_ndcg = 0
    t_ncrr = 0
    num_batches = len(test_users) // test_batch_size + 1
    for batch_users in tqdm(utils.generate_batches(test_users, test_batch_size), total=num_batches):
        with torch.no_grad():
            allPos = dataset.getUserPosItems(batch_users)
            batch_users_gpu = torch.Tensor(batch_users).long().to(device)
            all_ratings = model.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            all_ratings[exclude_index, exclude_items] = -(1<<10)

            _, top_k_ratings = torch.topk(all_ratings, TOP_K, dim=1)
            top_k_ratings = top_k_ratings.cpu().numpy()
            del all_ratings
        
        r = []
        for i,u in enumerate(batch_users):
            pred = list(map(lambda x: x in test_data[u], top_k_ratings[i]))
            pred = np.array(pred).astype('float')
            r.append(pred)
        r = np.array(r).astype('float')

        t_recall += utils.calc_recall(r, test_data, TOP_K, batch_users)
        t_ndcg += utils.calc_ndcg(r, test_data, TOP_K, batch_users)
        t_ncrr += utils.calc_ncrr(r, test_data, TOP_K, batch_users)
    
    t_recall /= len(test_users)
    t_ndcg /= len(test_users)
    t_ncrr /= len(test_users)
    hm = utils.calc_hm(t_recall, t_ndcg, t_ncrr)

    print(f"Recall: {t_recall}\n" + f"NDCG: {t_ndcg}\n" + f"NCRR: {t_ncrr}\n" + f"HM: {hm}")
    print("-"*40)
    return hm
    return "hm_" + f"{hm:.5f}"[2:]

def generate_submission(model, dataset, filename):
    model.eval()
    num_users = dataset.n_user
    batch_size = 100
    num_batches = num_users // batch_size + 1
    rec_list = []
    final_file = open(filename, "w")
    print('Generating submission...')
    for i in tqdm(range(num_batches)):
        batch_users = list(range(i*batch_size,min((i+1)*batch_size,num_users)))
        batch_users_gpu = torch.Tensor(batch_users).long().to(device)
        rating = model.getUsersRating(batch_users_gpu)
        _, rating_K = torch.topk(rating, k=50)
        rating_K = rating_K.cpu().numpy() + 1
        for r_row in rating_K:
            row = " ".join(str(x) for x in r_row)
            final_file.write(row + '\n')
    
    final_file.close()

def main(args):
    dataset = args['dataset']
    model = args['model']
    optimiser = args['optimiser']
    filename = args['filename']

    for epoch in range(1,EPOCHS+1):
        start = time.time()

        train_info = run_train(dataset, model, optimiser)

        total_time = time.time() - start
        output_info = f'[Time taken: {total_time:.1f}s]'
        print(f'Epoch[{epoch}/{EPOCHS}]   {train_info}  {output_info}')

        if epoch % TEST_INTERVAL == 0:
            if ALL_TRAIN:
                generate_submission(model, dataset, f"submissions/final_{epoch}_2_sub.txt")
            else:
                hm = run_test(dataset, model)
                if hm > 0.043:
                    str_hm = "hm_" + f"{hm:.5f}"[2:]
                    temp = f"{filename}_e{epoch}_{str_hm}"
                    torch.save(model.state_dict(), f"checkpoints/{temp}.pth")
                    generate_submission(model, dataset, f"submissions/{temp}_sub.txt")
    #run_test()
    #generate_submission(model, dataset, f"submissions/{'final_imp'}_sub.txt")


main(args1)