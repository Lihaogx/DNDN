from torch_geometric.nn import MessagePassing


class EdgeNeighborConv(MessagePassing):
    def __init__(self):
        super(EdgeNeighborConv, self).__init__(aggr='add')

    def forward(self, emb, edge_index, filt):
        return self.propagate(edge_index, size=(emb.size(0), emb.size(0)), x=emb, node_features=filt[edge_index[0, :]], emb=emb)

    def message(self, x_j, node_features):
        return x_j * node_features

    def update(self, aggr_out, emb):

        return aggr_out + emb/2