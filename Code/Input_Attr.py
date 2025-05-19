import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from Dataset import MEGA_DRP_Dataset
from Model import MEGA_DRP_Model
import Config
from tqdm.auto import tqdm
from captum.attr import IntegratedGradients
import pickle

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)

device = Config.device
torch.cuda.set_device(Config.device)

batch_size = 1


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


model_names = [
    "2l_696g_32bs_2l_8h_8cdh_0kf_e53",
    "2l_696g_32bs_2l_8h_8cdh_1kf_e73",
    "2l_696g_32bs_2l_8h_8cdh_2kf_e73",
    "2l_696g_32bs_2l_8h_8cdh_3kf_e59",
    "2l_696g_32bs_2l_8h_8cdh_4kf_e67",
]

cell = "Lung"
drug = "Afatinib"
interpret_set = MEGA_DRP_Dataset("interpret", cell=cell, drug=drug)
interpret_loader = DataLoader(interpret_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=batch_processing)
print(len(interpret_set))

lists = []
file_path = "../00 GDSC/exp_s.csv"
df = pd.read_csv(file_path, index_col=0)
row_means = df.mean(axis=1).tolist()

for k_fold, model_name in enumerate(model_names):

    model = MEGA_DRP_Model().to(device)
    model.load_state_dict(torch.load(f'../Model/{Config.symbol}/{model_name}.ckpt', map_location='cuda:0'))

    ig = IntegratedGradients(model)
    attrs = []

    model.eval()

    for batch in tqdm(interpret_loader):

        b_response_id, b_cell_id, b_drug_id, b_gene_indexes, b_exp, b_mut, b_cnv, b_adj_mat, b_v_feature, b_label = batch

        gene_indexes = torch.LongTensor(b_gene_indexes).to(device)
        exp = torch.FloatTensor(b_exp).to(device)
        mut = torch.FloatTensor(b_mut).to(device)
        cnv = torch.FloatTensor(b_cnv).to(device)
        adj_mat = torch.LongTensor(np.array(b_adj_mat))
        v_feature = torch.FloatTensor(np.array(b_v_feature))

        with torch.no_grad():
            logits = model(exp, mut, cnv, gene_indexes, adj_mat, v_feature)

        baseline_exp = torch.FloatTensor([row_means]).to(device)
        baseline_mut = torch.FloatTensor([[0] * Config.n_gene]).to(device)
        baseline_cnv = torch.FloatTensor([[0] * Config.n_gene]).to(device)

        ig_attr = ig.attribute((exp, mut, cnv), additional_forward_args=(gene_indexes, adj_mat, v_feature), baselines=(baseline_exp, baseline_mut, baseline_cnv))

        for b in range(batch_size):
            attr = (b_response_id[b],
                    b_cell_id[b],
                    b_drug_id[b],
                    logits.item(),
                    ig_attr[0][b].cpu().numpy(),
                    ig_attr[1][b].cpu().numpy(),
                    ig_attr[2][b].cpu().numpy()
                    )
            attrs.append(attr)

    with open(f"../Analysis/Input_Attr_Avg/input_attr_m{k_fold}_{cell}_{drug}.pkl", "wb") as f:
        pickle.dump(attrs, f)