import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, global_add_pool, GINConv
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_mean


class EdgeGCNConv(GCNConv):
    
    def __init__(self, in_channels, out_channels, edge_dim=1, **kwargs):
        super().__init__(in_channels + edge_dim, out_channels, **kwargs)
        self.edge_dim = edge_dim
        self.edge_weight_0 = nn.Parameter(torch.ones(1))
        self.edge_weight_1 = nn.Parameter(torch.ones(1))
        self.edge_transform = nn.Linear(1, 8)
        
    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        if edge_attr is not None:
            edge_mask = edge_attr.float().view(-1)
            
            edge_weights = torch.stack([self.edge_weight_0, self.edge_weight_1], dim=0)
            normalized_weights = F.softmax(edge_weights, dim=0)
            edge_weight = edge_weight * (edge_mask * normalized_weights[1] + 
                                    (1 - edge_mask) * normalized_weights[0])

            
            transformed_edge_features = F.leaky_relu(self.edge_transform(edge_attr.float().view(-1, 1)))
            
            node_edge_features = scatter_mean(
                transformed_edge_features,
                col,
                dim=0,
                dim_size=x.size(0)
            )
            
            x = torch.cat([x, node_edge_features], dim=1)
        
        edge_index, edge_weight = add_self_loops(
            edge_index, 
            edge_weight, 
            fill_value=edge_weight.mean().item()
        )
        
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

class EdgePredictor(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings, edge_index, edge_attr=None):
        row, col = edge_index
        source_embeddings = node_embeddings[row]
        target_embeddings = node_embeddings[col]
        
        edge_features = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        edge_predictions = self.edge_predictor(edge_features).squeeze()
        
        if edge_attr is not None:
            edge_labels = edge_attr.float().view(-1)
            edge_pred_loss = F.binary_cross_entropy(edge_predictions, edge_labels)
            return edge_predictions, edge_pred_loss
        
        return edge_predictions

class EdgeGATConv(GATConv):
    def __init__(self, in_channels, out_channels, edge_dim=1, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.edge_lin = nn.Linear(edge_dim, 1)
    

    def message(self, x_j, alpha, edge_attr=None):
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.softmax(alpha, self.node_dim)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        if edge_attr is not None:
            edge_weight = torch.sigmoid(self.edge_lin(edge_attr))
            alpha = alpha * edge_weight.view(-1)
            
        return x_j * alpha.unsqueeze(-1)


class HybridEdgeGATConv(GATConv):
    def __init__(self, in_channels, out_channels, edge_dim=1, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        
    def message(self, x_j, alpha, edge_attr=None):
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.softmax(alpha, self.node_dim)
        
        if edge_attr is not None:
            edge_mask = edge_attr.float().view(-1)
            alpha = alpha * (1.0 + 0.2 * edge_mask)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1) 


def build_conv(conv_type: str, use_edge_attr=False):
    
    if conv_type == "GCN":
        return EdgeGCNConv if use_edge_attr else GCNConv
    elif conv_type == "GIN":
        return lambda i, h: GINConv(
            nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
        )
    elif conv_type == "GAT":
        return EdgeGATConv if use_edge_attr else GATConv
    elif conv_type == "TransformerConv":
        return lambda i, h, **kwargs: TransformerConv(i, h, edge_dim=1) if use_edge_attr else TransformerConv(i, h)
    elif conv_type == "SAGE":
        return SAGEConv
    else:
        raise KeyError("[Model] GNN_TYPE can only be GAT, GCN, SAGE, GIN, and TransformerConv")


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=2, gnn_type="GCN", 
                 use_edge_attr=False, use_edge_predictor=False, edge_pred_weight=0.1):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        conv = build_conv(gnn_type, use_edge_attr)

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = torch.nn.LeakyReLU()

        self.pool = global_add_pool

        self.use_edge_predictor = use_edge_predictor
        self.edge_pred_weight = edge_pred_weight
        if use_edge_predictor:
            self.edge_predictor = EdgePredictor(hidden_dim)
        

        if n_layer < 1:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(n_layer))
        elif n_layer == 1:
            if self.use_edge_attr and gnn_type in ["GCN", "GAT", "TransformerConv"]:
                self.conv_layers = torch.nn.ModuleList([conv(input_dim, output_dim, edge_dim=1)])
            else:
                self.conv_layers = torch.nn.ModuleList([conv(input_dim, output_dim)])
        elif n_layer == 2:
            if self.use_edge_attr and gnn_type in ["GCN", "GAT", "TransformerConv"]:
                self.conv_layers = torch.nn.ModuleList([
                    conv(input_dim, hidden_dim, edge_dim=1),
                    conv(hidden_dim, output_dim, edge_dim=1)
                ])
            else:
                self.conv_layers = torch.nn.ModuleList([
                    conv(input_dim, hidden_dim),
                    conv(hidden_dim, output_dim)
                ])
        else:
            if self.use_edge_attr and gnn_type in ["GCN", "GAT", "TransformerConv"]:
                layers = [conv(input_dim, hidden_dim, edge_dim=1)]
                for _ in range(n_layer - 2):
                    layers.append(conv(hidden_dim, hidden_dim, edge_dim=1))
                layers.append(conv(hidden_dim, output_dim, edge_dim=1))
            else:
                layers = [conv(input_dim, hidden_dim)]
                for _ in range(n_layer - 2):
                    layers.append(conv(hidden_dim, hidden_dim))
                layers.append(conv(hidden_dim, output_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        device = x.device
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        if batch is not None:
            batch = batch.to(device)
            
        intermediate_embeddings = None

        for i, graph_conv in enumerate(self.conv_layers[0:-1]):
            if self.use_edge_attr and edge_attr is not None:
                x = graph_conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = graph_conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, training=self.training)

            if i == len(self.conv_layers) - 2:
                intermediate_embeddings = x

        if self.use_edge_attr and edge_attr is not None:
            node_emb = self.conv_layers[-1](x, edge_index, edge_attr=edge_attr)
        else:
            node_emb = self.conv_layers[-1](x, edge_index)
        
        edge_pred_loss = torch.tensor(0.0, device=device)
        
        if self.use_edge_predictor and edge_attr is not None and intermediate_embeddings is not None:
            try:
                _, edge_pred_loss = self.edge_predictor(intermediate_embeddings, edge_index, edge_attr)
            except Exception as e:
                print(f"Warning: Edge prediction error: {e}")
                edge_pred_loss = torch.tensor(0.0, device=device)

        if batch is None:
            if self.use_edge_predictor:
                return node_emb, edge_pred_loss
            return node_emb

        device = batch.device
        ones = torch.ones_like(batch).to(device)
        nodes_per_graph = global_add_pool(ones, batch)
        cum_num = torch.cat((torch.LongTensor([0]).to(device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))

        graph_emb = self.pool(node_emb, batch)

        if self.use_edge_predictor:
            return node_emb[cum_num], graph_emb, edge_pred_loss

        return node_emb[cum_num], graph_emb

class PromptLinearNet(nn.Module):
    def __init__(self, hidden_dim, threshold=0.1) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.threshold = threshold

    def forward(self, ego_nodes, central_nodes):
        similarity = torch.sum(ego_nodes * central_nodes, dim=1, keepdim=True)
        pred_logits = self.predictor(similarity)
        return pred_logits

    def make_prediction(self, ego_nodes, central_nodes):
        similarity = torch.sum(ego_nodes * central_nodes, dim=1, keepdim=True)
        pred_logits = self.predictor(similarity).squeeze(1)
        pos = torch.where(pred_logits >= self.threshold, 1.0, 0.0)
        return pos.nonzero().t().squeeze(0).detach().cpu().numpy().tolist()
