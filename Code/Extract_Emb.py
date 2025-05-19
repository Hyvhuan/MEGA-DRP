import torch
from Model import MEGA_DRP_Model
import Config

device = Config.device
model = MEGA_DRP_Model().to(device)

model_names = [
    "649g_32bs_8h_3l_8cdh_0kf_mix_e65",
    "649g_32bs_8h_3l_8cdh_1kf_mix_e69",
    "649g_32bs_8h_3l_8cdh_2kf_mix_e65",
    "649g_32bs_8h_3l_8cdh_3kf_mix_e57",
    "649g_32bs_8h_3l_8cdh_4kf_mix_e63",
]
model.load_state_dict(torch.load(f"Model/mix/{model_names[0]}.ckpt", map_location="cuda:0"))

# # 提取 embedding 层的权重
# embedding_weights = model.ceb.embedding.weight.data
#
# # 保存为 .pt 文件
# torch.save(embedding_weights, "embedding_weights.pt")
#
# # 保存为 numpy 数组
# embedding_weights_numpy = embedding_weights.cpu().numpy()
# np.save("embedding_weights.npy", embedding_weights_numpy)

# # 假设已经加载模型，提取 embedding 层权重
embedding_weights = model.ceb.embedding.weight.data.cpu().numpy()  # (649, 64)
#
# # 使用 t-SNE 降维
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# embedding_2d = tsne.fit_transform(embedding_weights)  # (649, 2)
#
# # 可视化
# plt.figure(figsize=(10, 8))
# plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=10, cmap="viridis")
#
# # 添加标签（可选，假设每个索引有意义）
# for i, point in enumerate(embedding_2d):
#     plt.text(point[0], point[1], str(i), fontsize=8, alpha=0.7)
#
# plt.title("t-SNE Visualization of Embedding Layer")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.grid(True)
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# 提取嵌入层权重
embedding_weights = model.ceb.embedding.weight.data.cpu().numpy()

# 使用 PCA 将维度从 64 降到 10
pca = PCA(n_components=5)
reduced_embedding = pca.fit_transform(embedding_weights)

# 使用降维后的数据进行聚类和热图可视化
sns.clustermap(
    reduced_embedding,
    cmap="viridis",
    method="ward",
    metric="euclidean",
    figsize=(12, 10)
)
plt.title("Clustered Heatmap with PCA Reduced Embedding")
plt.show()