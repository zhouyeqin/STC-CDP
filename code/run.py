from _heapq import heapify, heappop
from collections import defaultdict

import networkx as nx

import data
import metrics
import pretrain
import utils
import argparse
import torch
import model
import numpy as np
import time
import os
import math
from torch_geometric.utils import k_hop_subgraph
import torch
from datetime import datetime

from STC import gb_based_alg

if __name__ == "__main__":
    print('= ' * 20)
    print('## Starting Time:', utils.get_cur_time(), flush=True)
    total_start_time = time.time()

    parser = argparse.ArgumentParser(description="ProCom")

    parser.add_argument("--dataset", type=str, default="facebook")
    parser.add_argument("--seeds", type=int, default=[0,1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--pretrain_method", type=str, default="ProCom")
    parser.add_argument("--pretrain_epoch", type=int, default=15)
    parser.add_argument("--prompt_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--from_scratch", type=int, default=1)
    parser.add_argument("--node_scale", type=float, default=1.0)
    parser.add_argument("--subg_scale", type=float, default=0.01)

    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max_subgraph_size", type=int, default=20)
    parser.add_argument("--num_shot", type=int, default=2)
    parser.add_argument("--num_pred", type=int, default=1000)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--generate_k", type=int, default=2)

    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_type", type=str, default="GAT")

    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument('--use_edge_attr', default=True, help='Whether to use edge attributes in GNN')
    parser.add_argument('--use_edge_predictor', default=True, action='store_true', help='Whether to use edge prediction as auxiliary task')
    parser.add_argument('--edge_pred_weight', type=float, default=3, help='Weight for edge prediction loss')
    parser.add_argument('--use_prompt', type=int, default=0, help='Whether to perform prompt tuning (1: yes, 0: no)')
    args = parser.parse_args()

    if args.dataset == "facebook":
        args.num_pred = 200
    elif args.dataset in ["dblp", "amazon", "twitter"]:
        args.num_pred = 5000

    print(args)
    print("\n")

    data_load_start = time.time()
    num_node, num_edge, num_community, graph_data, nx_graph, communities = data.prepare_data(args.dataset)
    data_load_time = time.time() - data_load_start
    print(f"Finish loading data: {graph_data}")
    print(f"Data loading time: {data_load_time:.2f} seconds\n")

    

    pretrain_start = time.time()
    input_dim = graph_data.x.size(1)
    print("Perform pre-training ... ")
    print(f"GNN Configuration gnn_type({args.gnn_type}), num_layer({args.n_layers}), hidden_dim({args.hidden_dim})")
    device = torch.device(args.device)
    utils.set_seed(args.seeds[0])
    pretrain_model = pretrain.PreTrain(dataset=args.dataset,
                                       gnn_type=args.gnn_type,
                                       input_dim=input_dim,
                                       hidden_dim=args.hidden_dim,
                                       num_layers=args.n_layers,
                                       use_edge_attr=args.use_edge_attr,
                                       use_edge_predictor=args.use_edge_predictor,
                                       edge_pred_weight=args.edge_pred_weight,
                                       device=device)
    print(pretrain_model.gnn)
    num_pretrain_param = sum(p.numel() for p in pretrain_model.gnn.parameters())
    print(f"[Parameters] Number of parameters in GNN {num_pretrain_param}")

    pretrain_file_path = f"pretrain_models/{args.dataset}_{args.node_scale}_{args.subg_scale}_model.pt"
    if not args.from_scratch and os.path.exists(pretrain_file_path):
        pretrain_model.gnn.load_state_dict(torch.load(pretrain_file_path))
        print(f"Loading PRETRAIN-GNN file from {pretrain_file_path} !\n")
    else:
        if args.pretrain_method == "ProCom":
            print("Pretrain with ProCom proposed Dual-level Context-aware Loss ... ")
            pretrain_model.train_model(graph_data,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 epochs=args.pretrain_epoch,
                                 subg_max_size=args.max_subgraph_size,
                                 num_hop=args.k,
                                 node_scale=args.node_scale,
                                 subg_scale=args.subg_scale)
            torch.save(pretrain_model.gnn.state_dict(), pretrain_file_path)
        print(f"Pretrain Finish!\n")

    all_node_emb = pretrain_model.generate_all_node_emb(pretrain_model.gnn, graph_data.to(device))
    all_node_emb = all_node_emb.detach()
    pretrain_time = time.time() - pretrain_start
    print(f"Pre-training time: {pretrain_time:.2f} seconds\n")
    print("Pre-processing for K-EGO-NET extraction")
    node2ego_mapping = []
    if os.path.exists(f"../data/{args.dataset}/{args.generate_k}-ego.txt"):
        with open(f"../data/{args.dataset}/{args.generate_k}-ego.txt", 'r') as file:
            for line in file.readlines():
                content = line.split(" ")[1:]
                node2ego_mapping.append([int(node) for node in content])
    else:
        for node in range(num_node):
            node_k_ego, _, _, _ = k_hop_subgraph(node, num_hops=args.generate_k, edge_index=graph_data.edge_index,
                                                 num_nodes=num_node)
            node_k_ego = node_k_ego.detach().cpu().numpy().tolist()
            node2ego_mapping.append(node_k_ego)
            if node % 5000 == 0:
                print(f"***pre-processing {node} finish")
    print("Pre-preocessing Finish!\n")

    num_single_match = int(args.num_pred / args.num_shot)
    all_scores = []
    all_times = []
    for j in range(args.run_times):
        run_start_time = time.time()
        print(f"Times {j}")
        utils.set_seed(args.seeds[j])

        split_start = time.time()
        random_idx = list(range(num_community))
        np.random.shuffle(random_idx)
        print(random_idx[:args.num_shot])
        train_comms, test_comms = [communities[idx] for idx in random_idx[:args.num_shot]], [communities[idx] for idx in
                                                                                             random_idx[args.num_shot:]]
        train_com_emb = pretrain_model.generate_target_community_emb(pretrain_model.gnn,
                                                                     train_comms,
                                                                     graph_data).detach()
        step_times = {
            'data_split': time.time() - split_start
        }

        if args.use_prompt == 1:
            prompt_start = time.time()
            prompt_model = model.PromptLinearNet(args.hidden_dim, threshold=args.threshold).to(device)
            loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(prompt_model.parameters(), lr=args.lr, weight_decay=0.00001)
            num_prompt_param = sum(p.numel() for p in prompt_model.parameters())
            print(f"[Parameters] Number of parameters in Prompt {num_prompt_param}")

            for epoch in range(args.prompt_epoch):
                epoch_start = time.time()
                prompt_model.train()
                optimizer.zero_grad()
                st_time = time.time()
                all_central_nodes, all_ego_nodes, all_labels = torch.FloatTensor().to(device), torch.FloatTensor().to(
                    device), torch.FloatTensor().to(device)

                for community in train_comms:
                    central_node, k_ego, label = utils.generate_prompt_tuning_data(community, graph_data, nx_graph,
                                                                                   args.generate_k)
                    central_node_emb = all_node_emb[central_node, :]
                    ego_node_emb = all_node_emb[k_ego, :]

                    all_labels = torch.cat((all_labels, label.to(device)), dim=0)
                    all_central_nodes = torch.cat((all_central_nodes, central_node_emb), dim=0)
                    all_ego_nodes = torch.cat((all_ego_nodes, ego_node_emb), dim=0)

                pred_logits = prompt_model(all_ego_nodes, all_central_nodes)

                pt_loss = loss_fn(pred_logits, all_labels)
                pt_loss.backward()
                optimizer.step()
                epoch_time = time.time() - epoch_start
                print("***epoch: {:04d} | PROMPT TUNING train_loss: {:.5f} | cost time {:.3}s".format(epoch, pt_loss,
                                                                                                      epoch_time))
            step_times['prompt_tuning'] = time.time() - prompt_start
            print(f"Prompt Tuning time: {step_times['prompt_tuning']:.2f} seconds\n")
        else:
            print("Skipping prompt tuning as use_prompt is set to 0\n")
            prompt_model = None

        filter_start = time.time()
        candidate_comms = []

        st_time = time.time()
        for node in range(num_node):
            node_k_ego = node2ego_mapping[node]
            assert node in node_k_ego

            if args.use_prompt == 1:
                final_pos = prompt_model.make_prediction(all_node_emb[node_k_ego, :],
                                                         all_node_emb[[node] * len(node_k_ego), :])
            else:
                final_pos = list(range(len(node_k_ego)))

            if len(final_pos) > 0:
                candidate = [node_k_ego[idx] for idx in final_pos]

                if node not in candidate:
                    candidate = [node] + candidate

                candidate_comms.append(candidate)
        print(f"Number of candidate communities: {len(candidate_comms)}")
        print(f"Finish Candidate Filtering, Cost Time {time.time() - st_time:.5}s!\n")

        if len(candidate_comms) == 0:
            print("No candidate communities found, skipping this iteration...")
            continue

        embed_start = time.time()
        candidate_com_embeds = None

        batch_size = args.batch_size if args.dataset not in ["lj", "twitter", "dblp"] else 32
        num_batch = math.ceil(len(candidate_comms) / batch_size)
        st_time = time.time()
        for i in range(num_batch):
            start, end = i * batch_size, min((i + 1) * batch_size, len(candidate_comms))
            tmp_emb = pretrain_model.generate_target_community_emb(pretrain_model.gnn, candidate_comms[start:end],
                                                                   graph_data)
            tmp_emb = tmp_emb.detach().cpu().numpy()
            if candidate_com_embeds is None:
                candidate_com_embeds = tmp_emb
            else:
                candidate_com_embeds = np.vstack((candidate_com_embeds, tmp_emb))
        print(f"Finish Candidate Embedding Computation, Cost Time {time.time() - st_time:.5}s!\n")

        train_com_emb = train_com_emb.detach().cpu().numpy()
        pred_comms = []
        for i in range(args.num_shot):
            query = train_com_emb[i, :]
            distance = np.sqrt(np.sum(np.asarray(query - candidate_com_embeds) ** 2, axis=1))

            sort_dic = list(np.argsort(distance))

            length = 0

            for idx in sort_dic:
                if length >= num_single_match:
                    break

                neighs = candidate_comms[idx]
                if neighs not in pred_comms:
                    pred_comms.append(neighs)
                    length += 1

        output_dir = "facebook"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pred_comms.txt")

        with open(output_path, "w") as f:
            for comm in pred_comms:
                f.write(" ".join(map(str, comm)) + "\n")
        print(f"Predicted communities saved to {output_path}")

        f1, jaccard = metrics.eval_scores(pred_comms, test_comms, tmp_print=True)
        all_scores.append([f1, jaccard])
        utils.pred_community_analysis(pred_comms)
        print("\n")
        del prompt_model

        run_total_time = time.time() - run_start_time
        all_times.append({
            'run_total': run_total_time,
            'steps': step_times
        })
        print(f"Total time for run {j}: {run_total_time:.2f} seconds\n")

    if len(all_scores) > 0:
        avg_scores = np.mean(np.array(all_scores), axis=0)
        std_scores = np.std(np.array(all_scores), axis=0)
        print(
            f"Overall F1 {avg_scores[0]:.4f}+-{std_scores[0]:.5f}, Overall Jaccard {avg_scores[1]:.4f}+-{std_scores[1]:.5f}")
    else:
        print("No valid results were obtained in any iteration.")

    if len(all_times) > 0:
        print("\n=== Time Statistics ===")
        print(f"Total program time: {time.time() - total_start_time:.2f} seconds")
        print("\nAverage times per run:")
        
        avg_times = {
            'data_split': 0,
            'prompt_tuning': 0,
            'candidate_filtering': 0,
            'embedding_computation': 0,
            'final_prediction': 0,
            'run_total': 0
        }
        
        for key in avg_times.keys():
            if key == 'run_total':
                avg_times[key] = np.mean([t['run_total'] for t in all_times])
            else:
                avg_times[key] = np.mean([t['steps'].get(key, 0) for t in all_times])
        
        print(f"Data splitting: {avg_times['data_split']:.2f} seconds")
        if args.use_prompt == 1:
            print(f"Prompt tuning: {avg_times['prompt_tuning']:.2f} seconds")
        print(f"Candidate filtering: {avg_times['candidate_filtering']:.2f} seconds")
        print(f"Embedding computation: {avg_times['embedding_computation']:.2f} seconds")
        print(f"Final prediction: {avg_times['final_prediction']:.2f} seconds")
        print(f"Average total time per run: {avg_times['run_total']:.2f} seconds")

    print('\n## Finishing Time:', utils.get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
