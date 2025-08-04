import numpy as np
import random
import torch
import datetime
import time
import pytz
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)




def drop_nodes(graph_data, aug_ratio=0.1):
    r
    node_num, edge_num = graph_data.x.size(0), graph_data.edge_index.size(1)

    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_drop = idx_perm[:drop_num]
    idx_non_drop = idx_perm[drop_num:]
    idx_non_drop.sort()

    idx_dict = {idx_non_drop[n]: n for n in list(range(idx_non_drop.shape[0]))}

    device = graph_data.edge_index.device

    edge_index = graph_data.edge_index.detach().cpu().numpy()
    edge_attr = graph_data.edge_attr.detach().cpu().numpy()

    loop_edge_index = []
    index_to_remove = []
    for n in range(edge_num):
        if (edge_index[0, n] not in idx_drop) and (edge_index[1, n] not in idx_drop):
            edge_index_temp = [
                idx_dict[edge_index[0, n]],
                idx_dict[edge_index[1, n]]
            ]
            loop_edge_index.append(edge_index_temp)
        else:
            index_to_remove.append(n)
    edge_attr = np.delete(edge_attr, index_to_remove)

    try:
        new_edge_index = torch.tensor(loop_edge_index).transpose_(0, 1).to(device)
        new_edge_attr = torch.tensor(edge_attr).to(device)
        new_x = graph_data.x[idx_non_drop]
        new_graph_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    except:
        new_graph_data = graph_data
    return new_graph_data


def drop_edges(graph_data, aug_ratio=0.1):
    r
    edge_num = graph_data.edge_index.size(1)
    permute_num = int(edge_num * aug_ratio)
    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    new_edge_index = graph_data.edge_index[:, idx_delete]
    new_edge_attr = graph_data.edge_attr[idx_delete]
    return Data(x=graph_data.x, edge_index=new_edge_index,edge_attr=new_edge_attr)


def generate_corrupt_graph_view(graph_data, aug_ratio=0.15):
    x = random.random()

    if x < 0.5:
        return drop_nodes(graph_data, aug_ratio)
    return drop_edges(graph_data, aug_ratio)


def generate_prompt_tuning_data(train_comm, graph_data, nx_graph, k=2):
    
    degrees = [nx_graph.degree[node] for node in train_comm]
    sum_val = sum(degrees)
    degrees = [d / sum_val for d in degrees]
    central_node = np.random.choice(train_comm, 1, p=degrees).tolist()[0]

    k_ego_net, _, _, _ = k_hop_subgraph(central_node, num_hops=k, edge_index=graph_data.edge_index,
                                        num_nodes=graph_data.x.size(0))
    k_ego_net = k_ego_net.detach().cpu().numpy().tolist()

    labels = [[int(node in train_comm)] for node in k_ego_net]

    if 0 not in labels:
        random_negatives = np.random.choice(graph_data.x.size(0), len(k_ego_net)).tolist()
        random_negatives = [node for node in random_negatives if node not in k_ego_net]
        k_ego_net += random_negatives
        labels += [[0]] * len(random_negatives)
    return [central_node] * len(k_ego_net), k_ego_net, torch.FloatTensor(labels)


def pred_community_analysis(pred_comms):
    lengths = [len(com) for com in pred_comms]
    avg_length = np.mean(np.array(lengths))
    print(f"Predicted communitys #{len(pred_comms)}, avg size {avg_length:.4f}")
