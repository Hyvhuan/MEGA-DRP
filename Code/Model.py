import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
import numpy as np
import Config
import sqlite3
import pickle


class ExpScaling(nn.Module):
    def __init__(self):
        super(ExpScaling, self).__init__()

    def forward(self, emb, exp):

        # exp:[batch_size, n_gene] -> exp:[batch_size, n_gene, 1]
        exp = exp.unsqueeze(-1)

        # emb:[batch_size, n_gene, d_gene] = emb:[batch_size, n_gene, d_gene] * exp:[batch_size, n_gene, 1]
        emb = emb * exp

        return emb


class MutEncoding(nn.Module):
    def __init__(self):
        super(MutEncoding, self).__init__()

        self.W = nn.Linear(1, Config.d_gene, bias=False)

    def forward(self, emb, mut):

        # mut:[batch_size, n_gene] -> mut:[batch_size, n_gene, 1]
        mut = mut.unsqueeze(-1)

        # mut:[batch_size, n_gene, 1] -> mut:[batch_size, n_gene, d_gene]
        mut = self.W(mut)

        # emb:[batch_size, n_gene, d_gene] = emb:[batch_size, n_gene, d_gene] + mut:[batch_size, n_gene, d_gene]
        emb = emb + mut

        return emb


class CnvEncoding(nn.Module):
    def __init__(self):
        super(CnvEncoding, self).__init__()

        self.W = nn.Linear(1, Config.d_gene, bias=False)

    def forward(self, emb, cnv):

        # cnv:[batch_size, n_gene] -> cnv:[batch_size, n_gene, 1]
        cnv = cnv.unsqueeze(-1)

        # cnv:[batch_size, n_gene, 1] -> cnv:[batch_size, n_gene, d_gene]
        cnv = self.W(cnv)

        # emb:[batch_size, n_gene, d_gene] = emb:[batch_size, n_gene, d_gene] + cnv:[batch_size, n_gene, d_gene]
        emb = emb + cnv

        return emb


class CalculateAttentionWeights(nn.Module):
    def __init__(self):
        super(CalculateAttentionWeights, self).__init__()

    def forward(self, Q, K):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Config.d_k)

        # attn_weights: [batch_size, n_head, n_gene, n_gene/1]
        attn_weights = nn.Softmax(dim=-1)(scores)

        return attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(Config.d_gene, Config.d_k * Config.n_head, bias=False)
        self.W_K = nn.Linear(Config.d_gene, Config.d_k * Config.n_head, bias=False)
        self.W_V = nn.Linear(Config.d_gene, Config.d_v * Config.n_head, bias=False)
        self.caw = CalculateAttentionWeights()
        self.fc = nn.Linear(Config.d_v * Config.n_head, Config.d_gene, bias=False)
        self.ln = nn.LayerNorm(Config.d_gene)

    def forward(self, input_Q, input_K, input_V):

        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, Config.n_head, Config.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, Config.n_head, Config.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, Config.n_head, Config.d_v).transpose(1, 2)

        attn_weights = self.caw(Q, K)
        context = torch.matmul(attn_weights, V)

        concat_context = context.transpose(1, 2).reshape(batch_size, -1, Config.n_head * Config.d_v)
        output = self.fc(concat_context)

        output = self.ln(output + residual)

        return output


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(Config.d_gene, Config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_ff, Config.d_gene, bias=False)
        )
        self.ln = nn.LayerNorm(Config.d_gene)

    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        output = self.ln(output + residual)

        return output


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention()
        self.ff = FeedForward()

    def forward(self, inputs):

        output = self.self_attn(inputs, inputs, inputs)
        output = self.ff(output)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(Config.n_layer)])

    def forward(self, inputs):

        output = inputs
        for layer in self.layers:
            output = layer(output)

        return output


class GIN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            nn=nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU()
        ))

        self.conv2 = GINConv(
            nn=nn.Sequential(
                nn.Linear(d_hidden, d_out),
                nn.ReLU()
        ))

    def forward(self, edge_index, input_tensor):

        x = self.conv1(input_tensor, edge_index)
        x = self.conv2(x, edge_index)

        return x


class GAT(nn.Module):
    def __init__(self, d_in, d_hidden, n_head, d_out):
        super(GAT, self).__init__()

        self.conv1 = GATConv(d_in, d_hidden, heads=n_head, dropout=0)
        self.conv2 = GATConv(d_hidden * n_head, d_out, heads=1, concat=False, dropout=0)

    def forward(self, edge_index, input_tensor):

        x = F.elu(self.conv1(input_tensor, edge_index))
        x = self.conv2(x, edge_index)

        return x


class GCN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels=d_in, out_channels=d_hidden)
        self.conv2 = GCNConv(in_channels=d_hidden, out_channels=d_out)

    def forward(self, edge_index, input_tensor):

        x = F.relu(self.conv1(input_tensor, edge_index))
        x = self.conv2(x, edge_index)

        return x


class CD_Attention(nn.Module):
    def __init__(self):
        super(CD_Attention, self).__init__()

        self.W_Q = nn.Linear(Config.d_gene, Config.d_k * Config.cd_n_head, bias=False)
        self.W_K = nn.Linear(Config.d_gene, Config.d_k * Config.cd_n_head, bias=False)
        self.W_V = nn.Linear(Config.d_gene, Config.d_v * Config.cd_n_head, bias=False)
        self.caw = CalculateAttentionWeights()
        self.fc = nn.Linear(Config.d_v * Config.cd_n_head, Config.d_gene, bias=False)
        self.ln = nn.LayerNorm(Config.d_gene)

    def forward(self, input_Q, input_K, input_V):

        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, Config.cd_n_head, Config.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, Config.cd_n_head, Config.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, Config.cd_n_head, Config.d_v).transpose(1, 2)

        A = self.caw(Q, K)
        context = torch.matmul(A, V)

        concat_context = context.reshape(batch_size, -1, Config.d_v * Config.n_head)

        Z_ = self.fc(concat_context)

        Z = self.ln(Z_ + residual)

        # output:[batch_size, 1, d_gene]
        # A:[batch_size, n_head, 1, n_gene]
        # V:[batch_size, n_head, n_gene, d_gene]
        # context:[batch_size, n_head, 1, d_gene]
        # concat_context:[batch_size, 1, n_head * d_gene]
        # Z_:[batch_size, 1, d_gene]
        return Z, A, V, context, Z_
        # return Z


class CD_PM(nn.Module):
    def __init__(self):
        super(CD_PM, self).__init__()

        self.cd_attn = CD_Attention()

        self.fc = nn.Sequential(
            nn.Linear(Config.d_gene, Config.d_cdff1, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_cdff1, Config.d_cdff2, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_cdff2, Config.d_cdff3, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_cdff3, Config.n_output, bias=False),
        )

    def forward(self, cancer_representation, drug_representation):

        output, _, _, _, _ = self.cd_attn(drug_representation, cancer_representation, cancer_representation)
        # output = self.cd_attn(drug_representation, cancer_representation, cancer_representation)
        output = self.fc(output).squeeze(dim=1)

        return output


class MLP_PM(nn.Module):
    def __init__(self):
        super(MLP_PM, self).__init__()

        self.fc1 = nn.Linear(Config.d_gene, 1, bias=False)

        self.fc2 = nn.Sequential(
            nn.Linear(Config.n_gene + Config.d_v, Config.d_cdff1, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_cdff1, Config.d_cdff2, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_cdff2, Config.d_cdff3, bias=False),
            nn.ReLU(),
            nn.Linear(Config.d_cdff3, Config.n_output, bias=False),
        )

    def forward(self, cancer_representation, drug_representation):

        fc1_output = self.fc1(cancer_representation)
        tensor1 = fc1_output.squeeze(dim=2)

        tensor2 = drug_representation.squeeze(dim=1)

        concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)

        output = self.fc2(concatenated_tensor)

        return output


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

        self.embedding = nn.Embedding(Config.n_gene, Config.d_gene)

        if Config.pretrained_emb:

            conn = sqlite3.connect(f"../Data/G{Config.n_gene}/gene.db")
            cursor = conn.cursor()
            cursor.execute("SELECT id, emb FROM gene")
            rows = cursor.fetchall()
            conn.close()

            gene_emb_matrix = torch.empty((Config.n_gene, Config.d_gene))
            nn.init.xavier_uniform_(gene_emb_matrix)

            for gene_id, emb in rows:
                gene_id = int(gene_id)
                if emb is not None:
                    gene_emb_matrix[gene_id - 1] = torch.from_numpy(pickle.loads(emb))

            self.embedding.weight.data.copy_(gene_emb_matrix)

        self.exp_encoder = ExpScaling()
        self.mut_encoder = MutEncoding()
        self.cnv_encoder = CnvEncoding()

        self.encoder = Encoder()

    def forward(self, gene_indexes, exp, mut, cnv):

        # CE:
        # gene_index: [batch_size, n_gene]
        # gene_emb: [batch_size, n_gene, d_gene]
        gene_emb = self.embedding(gene_indexes)
        gene_emb = self.exp_encoder(gene_emb, exp)
        gene_emb = self.mut_encoder(gene_emb, mut)
        gene_emb = self.cnv_encoder(gene_emb, cnv)

        # cancer_representation: [batch_size, n_gene, d_gene]
        cancer_representation = self.encoder(gene_emb)

        return cancer_representation


class DE(nn.Module):
    def __init__(self):
        super(DE, self).__init__()

        if Config.gnn_type == "gin":
            self.gnn = GIN(Config.d_in, Config.d_hidden, Config.d_out)
        elif Config.gnn_type == "gat":
            self.gnn = GAT(Config.d_in, Config.d_hidden, Config.gat_n_head, Config.d_out)
        elif Config.gnn_type == "gcn":
            self.gnn = GCN(Config.d_in, Config.d_hidden, Config.d_out)

    def forward(self, adj_mat, v_feature):

        # DE:
        # drug_representation: [batch_size, 1, d_v]
        edge_index = []
        input_feature = np.empty((1, 1))
        offset = 0
        node_counts = []

        b_adj_mat = adj_mat.unbind(0)
        b_v_feature = v_feature.unbind(0)

        for g in zip(b_adj_mat, b_v_feature):

            adj_mat = np.array(g[0].cpu())
            v_feature = np.array(g[1].cpu())

            num_nodes = adj_mat.shape[0]
            node_counts.append(num_nodes)

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if adj_mat[i][j]:
                        edge_index.append([i + offset, j + offset])

            if offset == 0:
                input_feature = v_feature
            else:
                input_feature = np.vstack((input_feature, v_feature))

            offset += num_nodes

        edge_index = torch.LongTensor(edge_index).t().cuda()
        input_feature = torch.FloatTensor(input_feature).cuda()

        gnn_output = self.gnn(edge_index, input_feature)

        slices = torch.split(gnn_output, node_counts, dim=0)

        drug_representation_list = []
        for slice in slices:

            if Config.pooling_type:

                pooling_values, _ = torch.max(slice, dim=0)
                pooling_values = pooling_values.unsqueeze(0).unsqueeze(1)

            else:

                pooling_values = torch.mean(slice, dim=0)
                pooling_values = pooling_values.unsqueeze(0).unsqueeze(1)

            drug_representation_list.append(pooling_values)

        drug_representation = torch.cat(drug_representation_list, dim=0)

        return drug_representation


class PM(nn.Module):
    def __init__(self):
        super(PM, self).__init__()

        if Config.cd_attention:
            self.pm = CD_PM()
        else:
            self.pm = MLP_PM()

    def forward(self, cancer_representation, drug_representation):

        output = self.pm(cancer_representation, drug_representation)

        return output


class MEGA_DRP_Model(nn.Module):
    def __init__(self):
        super(MEGA_DRP_Model, self).__init__()

        self.ce = CE()
        self.de = DE()
        self.pm = PM()

    def forward(self, exp, mut, cnv, gene_indexes, adj_mat, v_feature):

        cancer_representation = self.ce(gene_indexes, exp, mut, cnv)
        drug_representation = self.de(adj_mat, v_feature)
        output = self.pm(cancer_representation, drug_representation)

        return output
