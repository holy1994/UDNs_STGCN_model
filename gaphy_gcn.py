import dgl
import torch
import torch.nn as nn
import dgl.function as fn

# 构建图，这里假设有3个节点，每条边有一个属性特征
edges = torch.tensor([0, 1, 1, 2])
edges = torch.stack([edges, torch.roll(edges, shifts=-1)]).t().contiguous()
g = dgl.graph(edges)
g.edata['edge_feat'] = torch.randn(g.number_of_edges(), 1)  # 边的属性特征

# 定义图神经网络模型
class EdgeGCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(EdgeGCN, self).__init__()
        self.conv = dgl.nn.GraphConv(in_feats, out_feats)
        self.fc = nn.Linear(out_feats, 1)

    def forward(self, g, node_feats, edge_feats):
        g.ndata['h'] = node_feats
        g.edata['w'] = edge_feats
        g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'h_neigh'))
        h_neigh = g.ndata['h_neigh']
        h_self = self.conv(g, node_feats)
        h = h_self + h_neigh
        return self.fc(h)

# 初始化模型
model = EdgeGCN(in_feats=1, out_feats=64)

# 模型输入
node_feats = torch.randn(g.number_of_nodes(), 1)  # 节点特征
edge_feats = g.edata['edge_feat']

# 模型前向传播
output = model(g, node_feats, edge_feats)
