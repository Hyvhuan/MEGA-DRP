import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import numpy as np
from Dataset import MEGA_DRP_Dataset
from Model import MEGA_DRP_Model
import Config
from tqdm.autonotebook import tqdm
from scipy.stats import pearsonr, spearmanr
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)

device = Config.device
torch.cuda.set_device(Config.device)

batch_size = Config.batch_size
n_epoch = Config.n_epoch


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


k_fold = int(os.path.basename(os.path.dirname(os.path.abspath(__file__)))[-1]) % 5
model_name = f"{Config.symbol}_" \
             f"{Config.n_gene}g_" \
             f"{Config.batch_size}bs_" \
             f"{Config.n_layer}l_" \
             f"{Config.n_head}h_" \
             f"{Config.cd_n_head}cdh_" \
             f"{k_fold}kf"

train_set = MEGA_DRP_Dataset("train", k_fold)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=batch_processing)
print(len(train_set))

valid_set = MEGA_DRP_Dataset("valid", k_fold)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=batch_processing)
print(len(valid_set))

model = MEGA_DRP_Model().to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-5)
best_loss = 99999

for epoch in range(n_epoch):

    model.train()

    train_loss = []
    train_preds = []
    train_labels = []

    for batch in tqdm(train_loader):

        b_response_id, b_cell_id, b_drug_id, b_gene_indexes, b_exp, b_mut, b_cnv, b_adj_mat, b_v_feature, b_label = batch

        gene_indexes = torch.LongTensor(b_gene_indexes).to(device)
        exp = torch.FloatTensor(b_exp).to(device)
        mut = torch.FloatTensor(b_mut).to(device)
        cnv = torch.FloatTensor(b_cnv).to(device)
        adj_mat = torch.LongTensor(np.array(b_adj_mat))
        v_feature = torch.FloatTensor(np.array(b_v_feature))
        label = torch.tensor(b_label).unsqueeze(0).transpose(0, 1).to(device)

        logits = model(exp, mut, cnv, gene_indexes, adj_mat, v_feature)

        loss = criterion(logits, label)

        optimizer.zero_grad()

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()

        acc = (logits.argmax(dim=-1) == label).float().mean()

        train_loss.append(loss.item())

        train_preds.append(logits.cpu().detach().numpy())
        train_labels.append(label.cpu().detach().numpy())

    train_loss = sum(train_loss) / len(train_loss)
    train_rmse = math.sqrt(train_loss)

    true_labels = np.vstack(train_labels).flatten()
    predictions = np.vstack(train_preds).flatten()
    train_pcc, _ = pearsonr(true_labels, predictions)
    train_scc, _ = spearmanr(true_labels, predictions)

    print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss = {train_loss:.5f}")
    print(f"[ Train ] RMSE = {train_rmse:.5f}")
    print(f"[ Train ] PCC = {train_pcc:.5f}")
    print(f"[ Train ] SCC = {train_scc:.5f}")

    model.eval()

    valid_loss = []
    valid_preds = []
    valid_labels = []

    for batch in tqdm(valid_loader):

        b_response_id, b_cell_id, b_drug_id, b_gene_indexes, b_exp, b_mut, b_cnv, b_adj_mat, b_v_feature, b_label = batch

        gene_indexes = torch.LongTensor(b_gene_indexes).to(device)
        exp = torch.FloatTensor(b_exp).to(device)
        mut = torch.FloatTensor(b_mut).to(device)
        cnv = torch.FloatTensor(b_cnv).to(device)
        adj_mat = torch.LongTensor(np.array(b_adj_mat)).to(device)
        v_feature = torch.FloatTensor(np.array(b_v_feature)).to(device)
        label = torch.tensor(b_label).unsqueeze(0).transpose(0, 1).to(device)

        with torch.no_grad():
            logits = model(exp, mut, cnv, gene_indexes, adj_mat, v_feature)

        loss = criterion(logits, label)

        acc = (logits.argmax(dim=-1) == label).float().mean()

        valid_loss.append(loss.item())

        valid_preds.append(logits.cpu().numpy())
        valid_labels.append(label.cpu().numpy())

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_rmse = math.sqrt(valid_loss)

    true_labels = np.vstack(valid_labels).flatten()
    predictions = np.vstack(valid_preds).flatten()
    valid_pcc, _ = pearsonr(true_labels, predictions)
    valid_scc, _ = spearmanr(true_labels, predictions)

    print(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}")
    print(f"[ Valid ] RMSE = {valid_rmse:.5f}")
    print(f"[ Valid ] PCC = {valid_pcc:.5f}")
    print(f"[ Valid ] SCC = {valid_scc:.5f}")

    if valid_loss < best_loss:
        print(f"Best model found at epoch {epoch + 1}\n")
        best_loss = valid_loss

    os.makedirs(f"../Model/{Config.symbol}", exist_ok=True)
    torch.save(model.state_dict(), f"../Model/{Config.symbol}/{model_name}_e{epoch + 1}.ckpt")

    os.makedirs(f"../Log/{Config.symbol}", exist_ok=True)
    with open(f"../Log/{Config.symbol}/{model_name}_log.txt", "a") as f:
        f.write(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss = {train_loss:.5f}\n")
        f.write(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}\n\n")