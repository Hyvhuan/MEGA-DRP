import torch
from torch.utils.data import DataLoader
import numpy as np
from Dataset import MEGA_DRP_Dataset
from Model import MEGA_DRP_Model
import Config
from tqdm.auto import tqdm
import sqlite3
import matplotlib.pyplot as plt
from sklearn.metrics import auc

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


predict_set = MEGA_DRP_Dataset("test_tcga")
predict_loader = DataLoader(predict_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=batch_processing)

conn = sqlite3.connect(Config.datasets[1])
epoch = list(range(1, 96))
result = conn.execute("""SELECT drug_id, drug, COUNT(*) as record_count
                         FROM response
                         WHERE drug_id != 196 AND drug_id != 38
                         GROUP BY drug
                         ORDER BY record_count DESC""")
drugs = result.fetchall()

for m in range(5):
    auc_values = {drug[0]: [] for drug in drugs}

    for e in epoch:
        model_name = f"2l_696g_32bs_2l_8h_8cdh_{m}kf_e{e}"
        model = MEGA_DRP_Model().to(device)
        model.load_state_dict(torch.load(f"../Model/2l/{model_name}.ckpt", map_location="cuda:0"))
        model.eval()

        for batch in tqdm(predict_loader, desc=f"Testing model {model_name}"):
            b_response_id, b_cell_id, b_drug_id, b_gene_indexes, b_exp, b_mut, b_cnv, b_adj_mat, b_v_feature, b_label = batch

            gene_indexes = torch.LongTensor(b_gene_indexes).to(device)
            exp = torch.FloatTensor(b_exp).to(device)
            mut = torch.FloatTensor(b_mut).to(device)
            cnv = torch.FloatTensor(b_cnv).to(device)
            adj_mat = torch.LongTensor(np.array(b_adj_mat))
            v_feature = torch.FloatTensor(np.array(b_v_feature))

            with torch.no_grad():
                logits = model(exp, mut, cnv, gene_indexes, adj_mat, v_feature)

            for i, y in enumerate(logits):
                conn.execute(f"UPDATE response SET pred_{m} = ? WHERE id = ?", (logits[i].item(), b_response_id[i]))
                conn.commit()

        for drug_id, drug, _ in drugs:
            result = conn.execute(f"SELECT pred_{m}, binary_response FROM response WHERE drug_id = ?", (drug_id,))
            rows = result.fetchall()

            y_prob = np.array([i[0] for i in rows])
            y = np.array([i[1] for i in rows])

            thresholds = np.linspace(min(y_prob), max(y_prob), 100)
            tpr_list = []
            fpr_list = []

            for threshold in thresholds:
                y_pred = (y_prob < threshold).astype(int)
                tp = np.sum((y_pred == 1) & (y == 1))
                fn = np.sum((y_pred == 0) & (y == 1))
                fp = np.sum((y_pred == 1) & (y == 0))
                tn = np.sum((y_pred == 0) & (y == 0))

                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)

                tpr_list.append(tpr)
                fpr_list.append(fpr)

            roc_auc = auc(fpr_list, tpr_list)
            auc_values[drug_id].append(roc_auc)

    with open(f"../AUC_TCGA/auc_values_model_{m}.txt", "w") as f:
        for drug_id, auc_list in auc_values.items():
            f.write(f"Drug ID {drug_id}: {','.join(map(str, auc_list))}\n")

    # fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharey="all")
    # count = 0
    #
    # for drug_id, drug, _ in drugs:
    #     i = count // 4
    #     j = count % 4
    #     count += 1
    #
    #     axes[i, j].plot(epoch, auc_values[drug_id], marker="o", color="b", label="AUC Trend")
    #     axes[i, j].set_xlabel("Epoch")
    #     axes[i, j].set_ylabel("AUC")
    #     axes[i, j].set_ylim(0, 1)
    #     axes[i, j].set_title(f"{drug}(model {m})", fontweight="bold")
    #     axes[i, j].legend(loc="lower right")
    #
    # plt.tight_layout()
    # plt.savefig(f"../AUC_TCGA/auc_trends_model_{m}.png", format="png")
    # plt.close()

