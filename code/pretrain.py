import time
import math
import torch
import torch.nn as nn
import random
from model import GNNEncoder
import torch.optim as optim
import data
import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.data import Data, Batch
import torch.nn.functional as F


class PreTrain(nn.Module):
    def __init__(self, dataset, gnn_type="GCN", input_dim=None, hidden_dim=None, num_layers=2,
                 use_edge_attr=False, use_edge_predictor=False, edge_pred_weight=0.1,
                 device=torch.device("cuda:0")):
        super(PreTrain, self).__init__()

        self.dataset = dataset
        self.device = device

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_layer = num_layers
        self.use_edge_predictor = use_edge_predictor
        self.edge_pred_weight = edge_pred_weight

        self.gnn = GNNEncoder(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              output_dim=hidden_dim,
                              n_layer=num_layers,
                              use_edge_attr=use_edge_attr,
                              use_edge_predictor=use_edge_predictor,
                              edge_pred_weight=edge_pred_weight,
                              gnn_type=gnn_type)
        self.gnn.to(device)

    def generate_all_candidate_community_emb(self, model, graph_data, batch_size=128, k=2, max_size=20):
        model.eval()
        node_num = graph_data.x.size(0)

        node_list = np.arange(0, node_num, 1)
        z = torch.Tensor(node_num, self.hidden_dim).to(self.device)
        group_nb = math.ceil(node_num / batch_size)

        for i in range(group_nb):
            maxx = min(node_num, (i + 1) * batch_size)
            minn = i * batch_size

            batch, _ = data.prepare_pretrain_data(node_list[minn:maxx].tolist(), data=graph_data, max_size=max_size,
                                                  num_hop=k)
            batch = batch.to(self.device)
            _, comms_emb = model(batch.x, batch.edge_index, batch.batch)
            z[minn:maxx] = comms_emb
            print(f"***Generate nodes embedding from idx {minn} to {maxx}")
        return z

    def generate_all_node_emb(self, model, graph_data):
        model.eval()

        result = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

        if isinstance(result, tuple):
            node_emb = result[0]
        else:
            node_emb = result

        return node_emb

    def generate_target_community_emb(self, model, comms, graph_data):
        batch = []
        num_nodes = graph_data.x.size(0)
        model.eval()
        for community in comms:
            if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                edge_index, edge_attr = subgraph(community, graph_data.edge_index, graph_data.edge_attr, relabel_nodes=True, num_nodes=num_nodes)
                g_data = Data(x=graph_data.x[community], edge_index=edge_index, edge_attr=edge_attr)
            else:
                edge_index, _ = subgraph(community, graph_data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
                g_data = Data(x=graph_data.x[community], edge_index=edge_index)
            batch.append(g_data)
        batch = Batch().from_data_list(batch).to(self.device)

        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            result = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            result = model(batch.x, batch.edge_index, None, batch.batch)

        if isinstance(result, tuple):
            if len(result) >= 2:
                comms_emb = result[1]
            else:
                comms_emb = result[0]
        else:
            comms_emb = result

        del batch
        return comms_emb

    def contrastive_loss(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()

        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)

        sim_matrix = torch.mm(x1_norm, x2_norm.t()) / T

        labels = torch.arange(batch_size, device=x1.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def train_subg(self, model, batch, optimizer, corrupt_batch=None, node_scale=1, subg_scale=0.1,
                   return_node_emb=False):
        model.train()
        old_params = {}
        for name, param in model.named_parameters():
            old_params[name] = param.clone().detach()

        batch = batch.to(self.device)

        if corrupt_batch is not None:
            corrupt_batch = corrupt_batch.to(self.device)

        if self.use_edge_predictor:
            z, summary, edge_pred_loss = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            z, summary = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            edge_pred_loss = torch.tensor(0.0, device=self.device)

        optimizer.zero_grad()

        loss = torch.tensor(0.0, device=self.device)

        if node_scale:
            node_loss = self.contrastive_loss(z, summary)
            loss += node_scale * node_loss
            print(f"Node loss: {node_loss.item():.5f}")

        if subg_scale and corrupt_batch:
            if self.use_edge_predictor:
                corrupt_result = model(corrupt_batch.x, corrupt_batch.edge_index,
                                       corrupt_batch.edge_attr, corrupt_batch.batch)

                if isinstance(corrupt_result, tuple) and len(corrupt_result) == 3:
                    corrupt_z, corrupt_summary, _ = corrupt_result
                else:
                    corrupt_z, corrupt_summary = corrupt_result
            else:
                corrupt_z, corrupt_summary = model(corrupt_batch.x, corrupt_batch.edge_index,
                                                   corrupt_batch.edge_attr, corrupt_batch.batch)
            subg_loss = self.contrastive_loss(summary, corrupt_summary)
            loss += subg_scale * subg_loss
            print(f"Subg loss: {subg_loss.item():.5f}")

        if self.use_edge_predictor:
            total_loss = loss + self.edge_pred_weight * 0.1 * edge_pred_loss
            print(f"Total loss: {loss.item():.5f}, Edge loss: {edge_pred_loss.item():.5f}")
        else:
            total_loss = loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        param_change = 0.0
        for name, param in model.named_parameters():
            change = torch.norm(param.data - old_params[name])
            param_change += change.item()

        print(f"Parameter change: {param_change:.5f}")

        if return_node_emb:
            return total_loss.item(), z
        return float(total_loss.detach().cpu().item())

    def train_model(self, graph_data, batch_size=128, lr=1e-3, decay=0.00001, epochs=100, subg_max_size=20, num_hop=1,
                    node_scale=1, subg_scale=0.1):
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        num_nodes = graph_data.x.size(0)

        for epoch in range(1, epochs + 1):
            st = time.time()

            node_list = random.sample(range(num_nodes), batch_size)
            batch_data, corrupt_batch_data = data.prepare_pretrain_data(node_list, data=graph_data,
                                                                        max_size=subg_max_size, num_hop=num_hop,
                                                                        corrupt=subg_scale)
            train_loss = self.train_subg(self.gnn, batch_data, optimizer, corrupt_batch=corrupt_batch_data,
                                         node_scale=node_scale, subg_scale=subg_scale)
            print(
                "***epoch: {:04d} | train_loss: {:.5f} | cost time {:.3}s".format(epoch, train_loss, time.time() - st))
