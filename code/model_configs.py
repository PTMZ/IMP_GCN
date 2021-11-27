from dataloader import RecsysData
from model import LightGCN, IMP_GCN
import torch

dataset = RecsysData()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")



model1 = IMP_GCN(dataset, latent_dim=300, n_layers=3, groups=3, dropout_bool=False, l2_w=0.0002, single=True).to(device)
optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
args1 = {"dataset": dataset, 
        "model": model1,
        "optimiser": optim1,
        "filename": f"imp_gcn_s_d{300}_l{3}_reg{2}_lr{'001'}"}

model2 = LightGCN(dataset, latent_dim=300, n_layers=3, dropout_bool=False, l2_w=1e-4).to(device)
optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
args2 = {"dataset": dataset,
        "model": model2,
        "optimiser": optim2,
        "filename": f"lgn_s_d{300}_l{3}_reg{35}"}

