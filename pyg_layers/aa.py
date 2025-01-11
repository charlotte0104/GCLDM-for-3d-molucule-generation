from graph_transformer_layer import GraphTransformerLayer
import torch


def get_fully_connected_edge_index(
        batch_index,
        node_mask = None
):
    adj = batch_index[:, None] == batch_index[None, :]
    edge_index = torch.stack(torch.where(adj), dim=0)
    if node_mask is not None:
        row, col = edge_index
        edge_mask = node_mask[row] & node_mask[col]
        edge_index = edge_index[:, edge_mask]
    return edge_index


model=GraphTransformerLayer(in_dim=6,hidden_dim=128,edge_dim=1,use_edges=True)

feats = torch.randn( 10, 6)
edge_attr=torch.randn(100,1)
index=torch.zeros(10)
edge_index=get_fully_connected_edge_index(index)


end=model(feats,edge_index,edge_attr)
print(end)