import torch
from torch.utils.data import DataLoader
import numpy as np
from Dataset import MEGA_DRP_Dataset
from Model import MEGA_DRP_Model
import Config
from tqdm.auto import tqdm
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


interpret_set = MEGA_DRP_Dataset("interpret")
interpret_loader = DataLoader(interpret_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=batch_processing)
print(len(interpret_set))

model_names = [
    "2l_696g_32bs_2l_8h_8cdh_0kf_e53",
    # "2l_696g_32bs_2l_8h_8cdh_1kf_e73",
    # "2l_696g_32bs_2l_8h_8cdh_2kf_e73",
    # "2l_696g_32bs_2l_8h_8cdh_3kf_e59",
    # "2l_696g_32bs_2l_8h_8cdh_4kf_e67",
]

lists = []

for k_fold, model_name in enumerate(model_names):

    b_info = []

    model = MEGA_DRP_Model().to(device)
    model.load_state_dict(torch.load(f'../Model/{Config.symbol}/{model_name}.ckpt', map_location='cuda:0'))

    def hook(module, fea_in, fea_out):
        global b_response_id, b_cell_id, b_drug_id, b_info

        # Z:[batch_size, 1, d_gene]
        # A:[batch_size, n_head, 1, n_gene]
        # V:[batch_size, n_head, n_gene, d_gene]
        # context:[batch_size, n_head, 1, d_gene]
        # Z_:[batch_size, 1, d_gene]
        b_Z = fea_out[0]
        b_A = fea_out[1]
        b_V = fea_out[2]
        b_context = fea_out[3]
        b_Z_ = fea_out[4]

        for response_id, cell_id, drug_id, Z, A, V, context, Z_ in zip(b_response_id, b_cell_id, b_drug_id, b_Z, b_A, b_V, b_context, b_Z_):
            b_info.append((response_id, cell_id, drug_id, Z))

        return None

    hook_handle = model.pm.pm.cd_attn.register_forward_hook(hook)

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

    hook_handle.remove()

    with open(f"../Analysis/Z/Z_m{k_fold}.pkl", "wb") as f:
        pickle.dump(b_info, f)
