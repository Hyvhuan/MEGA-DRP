import torch
from torch.utils.data import DataLoader
import numpy as np
from Dataset import MEGA_DRP_Dataset
from Model import MEGA_DRP_Model
import Config
from tqdm.auto import tqdm
from captum.attr import NeuronConductance
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
interpret_loader = DataLoader(interpret_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=batch_processing)
print(len(interpret_set))

model_names = [
    "649g_32bs_8h_3l_8cdh_0kf_mix_e65",
    "649g_32bs_8h_3l_8cdh_1kf_mix_e69",
    "649g_32bs_8h_3l_8cdh_2kf_mix_e65",
    "649g_32bs_8h_3l_8cdh_3kf_mix_e57",
    "649g_32bs_8h_3l_8cdh_4kf_mix_e63",
]
model = MEGA_DRP_Model().to(device)

lists = []

for m in range(5):

    model.load_state_dict(torch.load(f"Model/mix/{model_names[m]}.ckpt", map_location="cuda:0"))

    print(model)

    nc = NeuronConductance(model, model.pm.pm.cd_attn)
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

        nc_attr = nc.attribute((exp, mut, cnv), additional_forward_args=(gene_indexes, adj_mat, v_feature), target=0, neuron_selector=(0, 33))

        for b in range(batch_size):
            attr = (b_response_id[b],
                    b_cell_id[b],
                    b_drug_id[b],
                    nc_attr[0][b].detach().cpu().numpy(),
                    nc_attr[1][b].detach().cpu().numpy(),
                    nc_attr[2][b].detach().cpu().numpy(),
                    )
            attrs.append(attr)

    with open(f"neuron_attr_m{m}.pkl", "wb") as f:
        pickle.dump(attrs, f)