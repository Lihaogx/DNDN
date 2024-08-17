import numpy as np
import torch
import torch.nn.functional as F
from model.layer.gat_conv import GATConv

class SourceSinkEmbLayer(torch.nn.Module):
    def __init__(self, dropout, hidden_dim=32, combine='add',
                 new_node_feat=True, use_edge_attn=True):
        super(SourceSinkEmbLayer, self).__init__()
        self.source_conv = GATConv(hidden_dim, hidden_dim, double_input=False, concat=False, new_node_feat=new_node_feat, use_edge_attn=use_edge_attn)
        self.sink_conv = GATConv(hidden_dim, hidden_dim, double_input=False, concat=False, new_node_feat=new_node_feat, use_edge_attn=use_edge_attn)
        self.combine = combine
        self.dropout = dropout
        
    def forward(self, source_emb, sink_emb, source_edge_index, sink_edge_index):
        combined_emb = self.combine_emb(source_emb, sink_emb)
        
        new_source_emb = F.dropout(combined_emb, p=self.dropout, training=self.training)
        new_source_emb = self.source_conv(source_emb, source_edge_index)
        new_source_emb = F.prelu(source_emb, weight=torch.tensor(0.1).cuda())
        
        new_sink_emb = F.dropout(combined_emb, p=self.dropout, training=self.training)
        new_sink_emb = self.sink_conv(sink_emb, sink_edge_index)
        new_sink_emb = F.prelu(sink_emb, weight=torch.tensor(0.1).cuda())
        return new_source_emb, new_sink_emb
        
        
        
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