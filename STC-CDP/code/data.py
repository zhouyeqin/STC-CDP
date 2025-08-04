from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
from torch_geometric.utils import k_hop_subgraph, subgraph
import torch
import utils


def load_data(dataset_name):
    r
    communties = open(f"./data/{dataset_name}/{dataset_name}-1.90.cmty.txt")
    edge_labels = open(f"./data/{dataset_name}/{dataset_name}-1.90.edges_labels_base.txt")

    communties = [[int(i) for i in x.split()] for x in communties]
    labels = [[int(i) for i in l.split()] for l in edge_labels]
    edges = [edge[:2] for edge in labels]

    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communties = [[mapping[node] for node in com] for com in communties]

    num_node, num_edges, num_comm = len(nodes), len(edges), len(communties)
    print(f"[{dataset_name.upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")

    node_feats = None

    return num_node, num_edges, num_comm, nodes, edges, communties, node_feats,labels

def feature_augmentation(nodes, edges, num_node, normalize=True, feat_type='AUG'):
    r
    if feat_type == "ONE":
        return np.ones([num_node, 1], dtype=np.float32)

    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    node_degree = [g.degree[node] for node in range(num_node)]

    feat_matrix = np.zeros([num_node, 5], dtype=np.float32)
    feat_matrix[:, 0] = np.array(node_degree).squeeze()

    for node in range(num_node):
        if len(list(g.neighbors(node))) > 0:
            neighbor_deg = feat_matrix[list(g.neighbors(node)), 0]
            feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)
    return feat_matrix, g


def prepare_data(dataset="amazon"):
    r
    num_node, num_edge, num_community, nodes, edges, communities, features,labels = load_data(dataset)
    features, nx_graph = feature_augmentation(nodes, edges, num_node)

    converted_edges = [[v, u] for u, v in edges]
    new_edges = edges + converted_edges
    index_dict = {tuple(x): i for i, x in enumerate(new_edges)}
    sorted_labels = sorted(labels, key=lambda x: index_dict[tuple(x[:2])])
    sorted_labels = sorted_labels+sorted_labels

    graph_data = Data(x=torch.FloatTensor(features),
                      edge_index=torch.LongTensor(np.array(new_edges)).transpose(0, 1),
                      edge_attr=torch.LongTensor(np.array(list(map(lambda x: x[2], sorted_labels)))))
    return num_node, num_edge, num_community, graph_data, nx_graph, communities


def prepare_pretrain_data(node_list, data: Data, max_size=25, num_hop=2, corrupt=0):
    r
    batch, corrupt_batch = [], []

    num_nodes = data.x.size(0)


    for node in node_list:
        node_set, _, _, _ = k_hop_subgraph(node_idx=node, num_hops=num_hop, edge_index=data.edge_index, num_nodes=num_nodes)
        if len(node_set) > max_size:
            node_set = node_set[torch.randperm(node_set.shape[0])][:max_size]
            node_set = torch.unique(torch.cat([torch.LongTensor([node]), torch.flatten(node_set)]))

        node_list = node_set.detach().cpu().numpy().tolist()
        seed_idx = node_list.index(node)

        if seed_idx != 0:
            node_list[seed_idx], node_list[0] = node_list[0], node_list[seed_idx]

        assert node_list[0] == node

        edge_index, edge_attr = subgraph(node_list, data.edge_index,data.edge_attr, relabel_nodes=True, num_nodes=num_nodes)
        node_x = data.x[node_list]
        g_data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr)
        batch.append(g_data)

        if corrupt:
            corrupt_data = utils.generate_corrupt_graph_view(g_data)
            corrupt_batch.append(corrupt_data)

    batch = Batch().from_data_list(batch)
    if corrupt:
        corrupt_batch = Batch().from_data_list(corrupt_batch)

    return batch, corrupt_batch
