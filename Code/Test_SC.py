import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import numpy as np
from Dataset import MEGA_DRP_Dataset
from Model import MEGA_DRP_Model
import Config
from tqdm.auto import tqdm
import sqlite3
from scipy.stats import pearsonr, spearmanr

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)

device = Config.device
torch.cuda.set_device(Config.device)

batch_size = Config.batch_size


def batch_processing(items):
    b_response_id = []
    b_cell_id = []
    b_drug_id = []
    b_gene_indexes = []
    b_exp = []
    b_mut = []
    b_cnv = []
    b_adj_mat = []
    b_v_feature = []
    b_label = []

    for i in items:
        response_id, cell_id, drug_id, gene_indexes, exp, mut, cnv, adj_mat, v_feature, label = i
        b_response_id.append(response_id)
        b_cell_id.append(cell_id)
        b_drug_id.append(drug_id)
        b_gene_indexes.append(gene_indexes)
        b_exp.append(exp)
        b_mut.append(mut)
        b_cnv.append(cnv)
        b_adj_mat.append(adj_mat)
        b_v_feature.append(v_feature)
        b_label.append(label)

    batch = (b_response_id, b_cell_id, b_drug_id, b_gene_indexes, b_exp, b_mut, b_cnv, b_adj_mat, b_v_feature, b_label)

    return batch


# 2l
model_names = [
    "2l_696g_32bs_2l_8h_8cdh_0kf_e53",
    "2l_696g_32bs_2l_8h_8cdh_1kf_e73",
    "2l_696g_32bs_2l_8h_8cdh_2kf_e73",
    "2l_696g_32bs_2l_8h_8cdh_3kf_e59",
    "2l_696g_32bs_2l_8h_8cdh_4kf_e67",
]

predict_set = MEGA_DRP_Dataset("test_sc")
predict_loader = DataLoader(predict_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=batch_processing)
print(len(predict_set))

conn = sqlite3.connect(Config.datasets[2])

rmse_list, pcc_list, scc_list = [], [], []

for k_fold, model_name in enumerate(model_names):

    model = MEGA_DRP_Model().to(device)
    model.load_state_dict(torch.load(f"../Model/2l/{model_names[k_fold]}.ckpt", map_location="cuda:0"))

    criterion = nn.MSELoss()

    test_loss = []
    test_preds = []
    test_labels = []

    model.eval()

    for batch in tqdm(predict_loader):

        b_response_id, b_cell_id, b_drug_id, b_gene_indexes, b_exp, b_mut, b_cnv, b_adj_mat, b_v_feature, b_label = batch

        gene_indexes = torch.LongTensor(b_gene_indexes).to(device)
        exp = torch.FloatTensor(b_exp).to(device)
        mut = torch.FloatTensor(b_mut).to(device)
        cnv = torch.FloatTensor(b_cnv).to(device)
        adj_mat = torch.LongTensor(np.array(b_adj_mat))
        v_feature = torch.FloatTensor(np.array(b_v_feature))
        label = torch.tensor(b_label).unsqueeze(0).transpose(0, 1).to(device)

        with torch.no_grad():
            logits = model(exp, mut, cnv, gene_indexes, adj_mat, v_feature)

        loss = criterion(logits, label)

        test_loss.append(loss.item())

        test_preds.append(logits.cpu().numpy())
        test_labels.append(label.cpu().numpy())

        for i, y in enumerate(logits):
            conn.execute(f"UPDATE response SET pred_{k_fold} = ? WHERE id = ?", (logits[i].item(), b_response_id[i]))
            conn.commit()

    test_loss = sum(test_loss) / len(test_loss)
    rmse = math.sqrt(test_loss)

    true_labels = np.vstack(test_labels).flatten()
    predictions = np.vstack(test_preds).flatten()
    pcc, _ = pearsonr(true_labels, predictions)
    scc, _ = spearmanr(true_labels, predictions)

    print(f'[ Test ] RMSE = {rmse:.5f}')
    print(f'[ Test ] PCC = {pcc:.5f}')
    print(f'[ Test ] SCC = {scc:.5f}')

    rmse_list.append(rmse)
    pcc_list.append(pcc)
    scc_list.append(scc)

rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list)
pcc_mean = np.mean(pcc_list)
pcc_std = np.std(pcc_list)
scc_mean = np.mean(scc_list)
scc_std = np.std(scc_list)

print(f'[ Final Test ] RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}')
print(f'[ Final Test ] PCC = {pcc_mean:.4f} ± {pcc_std:.4f}')
print(f'[ Final Test ] SCC = {scc_mean:.4f} ± {scc_std:.4f}')