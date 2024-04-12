import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Linear, MSELoss
from torch_geometric.nn import MessagePassing

import utils.wasserstein as gdw

from layer.gat_conv import GATConv
from sg2dgm.PersistenceImager import PersistenceImager as PersistenceImager_nograd


class DNDN(torch.nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, num_layers=5, dropout=0.2, combine='add', out_dim=4,
                 new_node_feat=True, use_edge_attn=True):
        super(DNDN, self).__init__()

        self.source_conv = torch.nn.ModuleList()
        self.sink_conv = torch.nn.ModuleList()

        self.source_in_conv = GATConv(in_dim, hidden_dim, concat=False, new_node_feat=new_node_feat, use_edge_attn=use_edge_attn)
        self.sink_in_conv = GATConv(in_dim, hidden_dim, concat=False, new_node_feat=new_node_feat, use_edge_attn=use_edge_attn)

        for _ in range(num_layers-1):
            self.source_conv.append(GATConv(hidden_dim, hidden_dim, double_input=True, concat=False, new_node_feat=new_node_feat, use_edge_attn=use_edge_attn))
            self.sink_conv.append(GATConv(hidden_dim, hidden_dim, double_input=True, concat=False, new_node_feat=new_node_feat, use_edge_attn=use_edge_attn))

        self.source_out_linear = Linear(hidden_dim * 2, 2)
        self.sink_out_linear = Linear(hidden_dim * 2, 2)
        self.dim1_out_linear = Linear(hidden_dim * 2, 2)

        self.source_neighbor_conv = EdgeNeighborConv()
        self.sink_neighbor_conv = EdgeNeighborConv()

        self.num_layers = num_layers
        self.dropout = dropout
        self.combine = combine
        self.out_dim = out_dim

        self.pers_imager_nograd = PersistenceImager_nograd(resolution=5)
        self.PI_loss = MSELoss()

    def forward(self, x, source_edge_index, sink_edge_index, PD=None, compute_loss=True, p=2, type='train'):
        source_emb = x.detach()
        sink_emb = x.detach()

        source_emb = F.dropout(source_emb, p=self.dropout, training=self.training)
        source_emb = self.source_in_conv(source_emb, source_edge_index)
        source_emb = F.prelu(source_emb, weight=torch.tensor(0.1).cuda())

        sink_emb = F.dropout(sink_emb, p=self.dropout, training=self.training)
        sink_emb = self.sink_in_conv(sink_emb, sink_edge_index)
        sink_emb = F.prelu(sink_emb, weight=torch.tensor(0.1).cuda())

        for i in range(self.num_layers-1):
            combined_emb = self.combine_emb(source_emb, sink_emb)

            source_emb = F.dropout(combined_emb, p=self.dropout, training=self.training)
            source_emb = self.source_conv[i](source_emb, source_edge_index)
            source_emb = F.prelu(source_emb, weight=torch.tensor(0.1).cuda())

            sink_emb = F.dropout(combined_emb, p=self.dropout, training=self.training)
            sink_emb = self.sink_conv[i](sink_emb, sink_edge_index)
            sink_emb = F.prelu(sink_emb, weight=torch.tensor(0.1).cuda())

        combined_emb = self.combine_emb(source_emb, sink_emb)

        source_emb = F.relu(self.source_out_linear(combined_emb))
        sink_emb = F.relu(self.sink_out_linear(combined_emb))
        final_emb_dim0 = (source_emb + sink_emb) / 2

        source_emb_dim1 = F.relu(self.source_neighbor_conv(combined_emb, source_edge_index, x))
        sink_emb_dim1 = F.relu(self.sink_neighbor_conv(combined_emb, sink_edge_index, x))
        combined_emb = source_emb_dim1 + sink_emb_dim1
        final_emb_dim1 = F.relu(self.dim1_out_linear(combined_emb))

        final_emb = torch.cat((final_emb_dim0, final_emb_dim1), dim=1)

        if compute_loss:
            if type == 'train':
                wd_dim0, _, _, _, _ = self.compute_PD_loss(final_emb_dim0, PD[:, :2], p=p, num_models=1, type='train')
                wd_dim1, _, _, _, _ = self.compute_PD_loss(final_emb_dim1, PD[:, 2:], p=p, num_models=1, type='train')

                return final_emb, wd_dim0 + wd_dim1, wd_dim0, wd_dim1
            else:
                wd_dim0, _, _, _, _ = self.compute_PD_loss(final_emb_dim0, PD[:, :2], p=p, num_models=1, type='test')
                wd_dim1, _, _, _, _ = self.compute_PD_loss(final_emb_dim1, PD[:, 2:], p=p, num_models=1, type='test')

                emb_PI0 = torch.tensor(self.pers_imager_nograd.transform(np.array(final_emb_dim0.detach().cpu())).reshape(-1)).cuda()
                emb_PI = torch.tensor(self.pers_imager_nograd.transform(np.array(final_emb.detach().cpu())).reshape(-1)).cuda()
                PI0 = torch.tensor(self.pers_imager_nograd.transform(np.array(PD[:, :2].detach().cpu())).reshape(-1)).cuda()
                PIE0 = self.PI_loss(PI0, emb_PI0)

                return final_emb, wd_dim0 + wd_dim1, wd_dim0, wd_dim1, PIE0, emb_PI
        else:
            emb_PI0 = torch.tensor(self.pers_imager_nograd.transform(np.array(final_emb_dim0.detach().cpu())).reshape(-1)).cuda()
            PI0 = torch.tensor(self.pers_imager_nograd.transform(np.array(PD[:, :2].detach().cpu())).reshape(-1)).cuda()
            PIE0 = self.PI_loss(PI0, emb_PI0)
            return final_emb, PIE0

    def compute_PD_loss(self, PD1, PD2, p=2, num_models=1, type='train'):
        loss = torch.FloatTensor([0]).cuda()
        loss_xy = torch.FloatTensor([0]).cuda()
        loss_xd = torch.FloatTensor([0]).cuda()
        loss_yd = torch.FloatTensor([0]).cuda()

        if type == 'train':
            temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance(PD1, PD2, order=p, enable_autodiff=True, num_models=num_models)
            loss += temp_loss
            loss_xy += wxy; loss_xd += wxd; loss_yd += wyd
        else:
            temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance_inference(PD1, PD2, order=p, enable_autodiff=True)
            loss += temp_loss
            loss_xy += wxy; loss_xd += wxd; loss_yd += wyd

        return loss, ind_tmp_test, loss_xy, loss_xd, loss_yd

    def combine_emb(self, source_emb, sink_emb):
        if self.combine == 'add':
            return source_emb + sink_emb
        elif self.combine == 'max':
            return torch.max(source_emb, sink_emb)
        elif self.combine == 'mean':
            return (source_emb + sink_emb) / 2
        elif self.combine == 'min':
            return torch.min(source_emb, sink_emb)
        else:
            raise AttributeError('No combine method named {}'.format(self.combine))

class EdgeNeighborConv(MessagePassing):
    def __init__(self):
        super(EdgeNeighborConv, self).__init__(aggr='add')

    def forward(self, emb, edge_index, filt):
        return self.propagate(edge_index, size=(emb.size(0), emb.size(0)), x=emb, node_features=filt[edge_index[0, :]], emb=emb)

    def message(self, x_j, node_features):
        return x_j * node_features

    def update(self, aggr_out, emb):

        return aggr_out + emb/2
