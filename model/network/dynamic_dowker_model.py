import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss, CrossEntropyLoss
import utils.wasserstein as gdw
from torchmetrics import Metric
from torch_geometric.nn import global_mean_pool
from model.layer.gat_conv import GATConv
from model.layer.embedding_layer import SourceSinkEmbLayer
from model.layer.message_passing_layer import EdgeNeighborConv
from sg2dgm.PersistenceImager import PersistenceImager as PersistenceImager_nograd

class DowkerMetrics(Metric):
    def __init__(self, dist_sync_on_step=False, loss_type='dim0'):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("wd", default=[], dist_reduce_fx="mean")
        self.add_state("pi", default=[], dist_reduce_fx="mean")
        self.add_state("accuracy", default=[], dist_reduce_fx="mean")
        self.loss_type = loss_type
        self.pers_imager_nograd = PersistenceImager_nograd(resolution=5)
        self.pi_loss = MSELoss()
        
        
    def split_pd(self, num_nodes_per_graph, pd):
        pd_list = []
        start_idx = 0
        for num_nodes in num_nodes_per_graph:
            end_idx = start_idx + num_nodes
            pd_list.append(pd[start_idx:end_idx])
            start_idx = end_idx
        return pd_list
    
    def update(self, batch):
        wd_d0 = []
        wd_d1 = []
        pi_d0 = []
        pi_d1 = []
        num_nodes_per_graph = torch.bincount(batch.batch)
        pd_ground_truth_dim0 = self.split_pd(num_nodes_per_graph, batch.barcode_ground[:, :2])
        pd_predict_dim0 = self.split_pd(num_nodes_per_graph, batch.pd_dim0)
        for i in range(len(pd_ground_truth_dim0)):
            wd_dim0, _, _, _, _ = self.compute_PD_loss(pd_ground_truth_dim0[i], pd_predict_dim0[i])
            wd_d0.append(wd_dim0)
            emb_pi0 = torch.tensor(self.pers_imager_nograd.transform(np.array(pd_predict_dim0[i].detach().cpu())).reshape(-1)).cuda()
            ground_pi0 = torch.tensor(self.pers_imager_nograd.transform(np.array(pd_ground_truth_dim0[i].detach().cpu())).reshape(-1)).cuda()
            pi_d0.append(self.pi_loss(ground_pi0, emb_pi0))
        wd = torch.mean(torch.cat(wd_d0))
        pi_d0 = [tensor.unsqueeze(0) if tensor.dim() == 0 else tensor for tensor in pi_d0]
        pi = torch.mean(torch.cat(pi_d0))

        if self.loss_type == 'dim1':
            pd_ground_truth_dim1 = self.split_pd(num_nodes_per_graph, batch.barcode_ground[:, 2:])
            pd_predict_dim1 = self.split_pd(num_nodes_per_graph, batch.pd_dim1)
            for i in range(len(pd_ground_truth_dim1)):
                wd_dim1, _, _, _, _ = self.compute_PD_loss(pd_ground_truth_dim1[i], pd_predict_dim1[i])
                wd_d1.append(wd_dim1)
                emb_pi1 = torch.tensor(self.pers_imager_nograd.transform(np.array(pd_predict_dim0[i].detach().cpu())).reshape(-1)).cuda()
                ground_pi1 = torch.tensor(self.pers_imager_nograd.transform(np.array(pd_ground_truth_dim0[i].detach().cpu())).reshape(-1)).cuda()
                pi_d1.append(self.pi_loss(ground_pi1, emb_pi1))
            wd = wd + torch.mean(torch.cat(wd_d1))
            pi_d1 = [tensor.unsqueeze(0) if tensor.dim() == 0 else tensor for tensor in pi_d1]
            pi = pi + torch.mean(torch.cat(pi_d1))
        self.wd.append(wd)
        self.pi.append(pi)
                
                
    def compute_PD_loss(self, PD1, PD2, num_models=1):
        loss = torch.FloatTensor([0]).cuda()
        loss_xy = torch.FloatTensor([0]).cuda()
        loss_xd = torch.FloatTensor([0]).cuda()
        loss_yd = torch.FloatTensor([0]).cuda()

        temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance(PD1, PD2, order=2, enable_autodiff=True, num_models=num_models)
        loss += temp_loss
        loss_xy += wxy; loss_xd += wxd; loss_yd += wyd

        return loss, ind_tmp_test, loss_xy, loss_xd, loss_yd
    
    def compute(self):
        return  {
            'wd': torch.mean(torch.stack(self.wd)),
            'pi': torch.mean(torch.stack(self.pi)),
            'accuracy': torch.mean(torch.stack(self.accuracy))
        }
        
class DNDN(torch.nn.Module):
    def __init__(self, num_classes, in_dim=1, hidden_dim=32, num_layers=5, dropout=0.2, combine='add', 
                 new_node_feat=True, use_edge_attn=True):
        super(DNDN, self).__init__()
                
        self.layers = torch.nn.ModuleList()
        
        self.source_initial_layer = Linear(in_dim, hidden_dim)
        self.sink_initial_layer = Linear(in_dim, hidden_dim)
        
        for _ in range(num_layers):
            self.layers.append(SourceSinkEmbLayer(dropout, hidden_dim, combine, new_node_feat, use_edge_attn))

        self.source_out_linear = Linear(hidden_dim , 2)
        self.sink_out_linear = Linear(hidden_dim , 2)
        self.dim1_out_linear = Linear(hidden_dim , 2)
        
        
        self.source_neighbor_conv = EdgeNeighborConv()
        self.sink_neighbor_conv = EdgeNeighborConv()
        self.num_layers = num_layers
        self.dropout = dropout
        self.combine = combine

    def forward(self, batch):
        x, source_edge_index, sink_edge_index =  batch.x, batch.source_edge_index, batch.sink_edge_index
        source_emb = x.detach()
        sink_emb = x.detach()
        
        source_emb = self.source_initial_layer(source_emb)
        sink_emb = self.sink_initial_layer(sink_emb)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SourceSinkEmbLayer):
                source_emb, sink_emb = layer(source_emb, sink_emb, source_edge_index, sink_edge_index)
        
        combined_emb = self.combine_emb(source_emb, sink_emb)
        source_emb = F.relu(self.source_out_linear(combined_emb))
        sink_emb = F.relu(self.sink_out_linear(combined_emb))
        final_emb_dim0 = (source_emb + sink_emb) / 2

        source_emb_dim1 = F.relu(self.source_neighbor_conv(combined_emb, source_edge_index, x))
        sink_emb_dim1 = F.relu(self.sink_neighbor_conv(combined_emb, sink_edge_index, x))
        combined_emb = source_emb_dim1 + sink_emb_dim1
        final_emb_dim1 = F.relu(self.dim1_out_linear(combined_emb))
        
        final_PD = torch.cat((final_emb_dim0, final_emb_dim1), dim=1)
            
            
        batch.pd_dim0 = final_emb_dim0
        batch.pd_dim1 = final_emb_dim1
        
        return batch

    
    def split_pd(self, num_nodes_per_graph, pd):
        pd_list = []
        start_idx = 0
        for num_nodes in num_nodes_per_graph:
            end_idx = start_idx + num_nodes
            pd_list.append(pd[start_idx:end_idx])
            start_idx = end_idx
        return pd_list

    def loss(self, batch, loss_type):
        loss_dim0 = []
        loss_dim1 = []
        num_nodes_per_graph = torch.bincount(batch.batch)
        pd_ground_truth_dim0 = self.split_pd(num_nodes_per_graph, batch.barcode_ground[:, :2])
        pd_predict_dim0 = self.split_pd(num_nodes_per_graph, batch.pd_dim0)
        for i in range(len(pd_ground_truth_dim0)):
            wd_dim0, _, _, _, _ = self.compute_PD_loss(pd_ground_truth_dim0[i], pd_predict_dim0[i])
            loss_dim0.append(wd_dim0)
        loss = torch.mean(torch.cat(loss_dim0))
        
        if loss_type == 'dim1':
            pd_ground_truth_dim1 = self.split_pd(num_nodes_per_graph, batch.barcode_ground[:, 2:])
            pd_predict_dim1 = self.split_pd(num_nodes_per_graph, batch.pd_dim1)
            for i in range(len(pd_ground_truth_dim1)):
                wd_dim1, _, _, _, _ = self.compute_PD_loss(pd_ground_truth_dim1[i], pd_predict_dim1[i])
                loss_dim1.append(wd_dim1)
            loss = loss + torch.mean(torch.cat(loss_dim1))

        return loss
        
        
    def compute_PD_loss(self, PD1, PD2, num_models=1):
        loss = torch.FloatTensor([0]).cuda()
        loss_xy = torch.FloatTensor([0]).cuda()
        loss_xd = torch.FloatTensor([0]).cuda()
        loss_yd = torch.FloatTensor([0]).cuda()

        temp_loss, ind_tmp_test, wxy, wxd, wyd = gdw.wasserstein_distance(PD1, PD2, order=2, enable_autodiff=True, num_models=num_models)
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