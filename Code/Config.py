seed = 121
device = 0
batch_size = 32
n_epoch = 300
lr = 0.0001

n_gene = 696
datasets = [f"../Data/G{n_gene}/gdsc.db",
            f"../Data/G{n_gene}/tcga.db",
            f"../Data/G{n_gene}/sc.db",
            f"../Data/G{n_gene}/pdx.db"]
blind = ["mix", "cb", "db"][0]
symbol = "mlp"

pretrained_emb = 0
d_gene = 64
n_head = 8
d_k = 64
d_v = 64
d_ff = 512
n_layer = 2

gnn_type = ["gin", "gat", "gcn"][0]
d_in = 78
d_hidden = 64
d_out = 64
gat_n_head = 1
pooling_type = 1

cd_attention = 0
# cd_attention = 1
cd_n_head = 8
d_cdff1 = 512
d_cdff2 = 128
d_cdff3 = 64
n_output = 1