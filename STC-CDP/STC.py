from collections import defaultdict
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import argparse

from Heap import build_heap

weak = 0
strong = 1

def draw_graph(G):
    pos = nx.spring_layout(G, seed=1)
    nx.draw_networkx(G, pos, node_color='green', edge_color='red', node_size=500, width=2)
    plt.show()

def load_data_STC(dataset_name):
    edge_labels = open(f"./data/{dataset_name}/{dataset_name}-1.90.ungraph.txt")
    labels = [[int(i) for i in l.split()] for l in edge_labels]
    edges = [edge[:2] for edge in labels]


    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    nx_graph = nx.Graph(edges)
    nx_graph.add_nodes_from(nodes)

    return nx_graph


def create_wedge_graph_org(G, max_wedges=0, max_edges=0):
    
    print(f"Starting to build wedge graph, original graph size: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    nodes = list(G.nodes)
    
    print("Precalculating node adjacency information...")
    adj_dic = {}
    
    total_nodes = len(nodes)
    batch_size = max(1, total_nodes // 100)
    
    for i in range(0, total_nodes, batch_size):
        batch_end = min(i + batch_size, total_nodes)
        batch_nodes = nodes[i:batch_end]
        
        for v in batch_nodes:
            adj_dic[v] = set(G.neighbors(v))
        
        progress = min(100, int((batch_end / total_nodes) * 100))
        print(f"Precalculation progress: {progress}%", end='\r')
    
    print("\nBuilding wedge structure...")
    wedge = defaultdict(set)
    edge_set = set()
    
    wedge_count = 0
    prev_percent = 0
    
    sorted_nodes = sorted(nodes, key=lambda n: len(adj_dic[n]), reverse=True)
    
    for idx, i in enumerate(sorted_nodes):
        ne_i = adj_dic[i]
        if len(ne_i) <= 1:
            continue
            
        for j in ne_i:
            ne_j = adj_dic[j]
            common_neighbors = ne_i.intersection(ne_j) - {i, j}
            
            for k in common_neighbors:
                if i < k:
                    wedge[j].add((i, k))
                    wedge_count += 1
                    
                    edge_set.add((min(i, j), max(i, j)))
                    edge_set.add((min(j, k), max(j, k)))
                    
                    if max_wedges > 0 and wedge_count >= max_wedges:
                        print(f"\nReached maximum wedge count limit ({max_wedges}), ending wedge construction early")
                        break
            
            if (max_wedges > 0 and wedge_count >= max_wedges) or \
               (max_edges > 0 and len(edge_set) >= max_edges):
                break
        
        if (max_wedges > 0 and wedge_count >= max_wedges) or \
           (max_edges > 0 and len(edge_set) >= max_edges):
            print(f"\nReached limit conditions (wedge count={wedge_count}, edge count={len(edge_set)}), ending wedge construction early")
            break
            
        current_percent = min(100, int((idx+1) / len(sorted_nodes) * 100))
        if current_percent > prev_percent:
            print(f"Wedge structure building progress: {current_percent}%, wedges found: {wedge_count}", end='\r')
            prev_percent = current_percent
    
    print(f"\nFound {wedge_count} wedge structures and {len(edge_set)} edges")
    
    print("Assigning IDs to edges and building wedge graph...")
    edge_dic = {edge: idx for idx, edge in enumerate(edge_set)}
    v_dic = {idx: edge for edge, idx in edge_dic.items()}
    
    wedge_edges = set()
    te_weak = {}
    te_add = {}
    
    batch_size = max(1, len(wedge) // 50)
    wedge_items = list(wedge.items())
    
    for batch_idx in range(0, len(wedge_items), batch_size):
        batch_end = min(batch_idx + batch_size, len(wedge_items))
        batch = wedge_items[batch_idx:batch_end]
        
        for j, pairs in batch:
            for (a, b) in pairs:
                a_j = (min(a, j), max(a, j))
                j_b = (min(j, b), max(j, b))
                
                if a_j in te_weak:
                    te_weak[a_j].append((j, b))
                else:
                    te_weak[a_j] = [(j, b)]
                
                a_b = (min(a, b), max(a, b))
                if a_b not in te_add:
                    te_add[a_b] = []
                te_add[a_b].append([(j, b), (j, a)])
                
                e1 = edge_dic[a_j]
                e2 = edge_dic[j_b]
                wedge_edges.add((e1, e2))
        
        progress = min(100, int((batch_end / len(wedge_items)) * 100))
        print(f"Wedge graph building progress: {progress}%", end='\r')
    
    print("\nCreating final wedge graph...")
    wedge_G = nx.Graph()
    wedge_G.add_edges_from(wedge_edges)
    
    isolates = list(nx.isolates(wedge_G))
    if isolates:
        print(f"Removing {len(isolates)} isolated nodes")
        wedge_G.remove_nodes_from(isolates)
    
    print(f'Wedge graph completed. V(W): {wedge_G.number_of_nodes()}  E(W): {wedge_G.number_of_edges()}')
    return [wedge_G, edge_dic, v_dic, te_weak, te_add]


def parallel_mvc_chunk(chunk_nodes, G, node_degrees):
    
    local_mvc = set()
    covered_edges = set()
    
    sorted_nodes = sorted(chunk_nodes, key=lambda n: node_degrees.get(n, 0), reverse=True)
    
    for node in sorted_nodes:
        if node_degrees.get(node, 0) == 0:
            continue
            
        adjacent_edges = set()
        for u, v in G.edges(node):
            if (u, v) not in covered_edges and (v, u) not in covered_edges:
                adjacent_edges.add((u, v))
                
        if adjacent_edges:
            local_mvc.add(node)
            covered_edges.update(adjacent_edges)
    
    return local_mvc, covered_edges

def minimum_vertex_cover_parallel(G, num_processes=None):
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        
    node_degrees = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    sorted_nodes = sorted(G.nodes(), key=lambda n: node_degrees[n], reverse=True)
    
    chunk_size = max(1, len(sorted_nodes) // num_processes)
    node_chunks = [sorted_nodes[i:i+chunk_size] for i in range(0, len(sorted_nodes), chunk_size)]
    
    mvc = set()
    remaining_edges = set(G.edges())
    
    for i in range(min(5, len(node_chunks))):
        if i < len(node_chunks):
            chunk = node_chunks[i]
            
            local_mvc, covered_edges = parallel_mvc_chunk(chunk, G, node_degrees)
            
            mvc.update(local_mvc)
            remaining_edges -= covered_edges
            
            if not remaining_edges:
                break
                
            for edge in covered_edges:
                u, v = edge
                if u in node_degrees:
                    node_degrees[u] = max(0, node_degrees[u] - 1)
                if v in node_degrees:
                    node_degrees[v] = max(0, node_degrees[v] - 1)
    
    if remaining_edges:
        subgraph = nx.Graph()
        subgraph.add_edges_from(remaining_edges)
        
        remaining_mvc = set()
        while remaining_edges:
            max_degree = -1
            max_node = None
            
            for node in subgraph.nodes():
                degree = subgraph.degree(node)
                if degree > max_degree:
                    max_degree = degree
                    max_node = node
            
            if max_node is None:
                break
                
            remaining_mvc.add(max_node)
            
            covered = list(subgraph.edges(max_node))
            remaining_edges -= set(covered)
            subgraph.remove_node(max_node)
            
        mvc.update(remaining_mvc)
    
    return mvc

def simple_minimum_vertex_cover(G):
    
    mvc = set()
    
    working_graph = G.copy()
    
    while working_graph.number_of_edges() > 0:
        max_degree = -1
        max_node = None
        
        for node in working_graph.nodes():
            degree = working_graph.degree(node)
            if degree > max_degree:
                max_degree = degree
                max_node = node
        
        if max_node is None or max_degree == 0:
            break
            
        mvc.add(max_node)
        
        working_graph.remove_node(max_node)
    
    return mvc


def fast_minimum_vertex_cover(G):
    
    mvc = set()
    
    remaining_edges = set(G.edges())
    if not remaining_edges:
        return mvc
    
    node_degrees = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    while remaining_edges:
        print(f"Remaining uncovered edges: {len(remaining_edges)}", end='\r')
        
        sorted_nodes = sorted(
            [n for n in G.nodes() if node_degrees[n] > 0],
            key=lambda n: node_degrees[n],
            reverse=True
        )
        
        if not sorted_nodes:
            break
            
        batch_size = max(1, len(sorted_nodes) // 10)
        high_degree_nodes = sorted_nodes[:batch_size]
        
        mvc.update(high_degree_nodes)
        
        edges_to_remove = set()
        for node in high_degree_nodes:
            for u, v in G.edges(node):
                if (u, v) in remaining_edges:
                    edges_to_remove.add((u, v))
                elif (v, u) in remaining_edges:
                    edges_to_remove.add((v, u))
                    
        remaining_edges -= edges_to_remove
        
        for edge in edges_to_remove:
            u, v = edge
            if u in node_degrees:
                node_degrees[u] = max(0, node_degrees[u] - 1)
            if v in node_degrees:
                node_degrees[v] = max(0, node_degrees[v] - 1)
    
    print()
    return mvc


def gb_based_alg(lam, G, isGdy=True, max_wedges=0, max_edges=0):
    
    edge_attr_dict = {edge: {"label": strong, "weight": 10} for edge in G.edges()}
    nx.set_edge_attributes(G, edge_attr_dict)
    
    [wedge_G, edge_dic, v_dic, te_weak, te_add] = create_wedge_graph_org(G, max_wedges, max_edges)
    
    wedge_size = wedge_G.number_of_nodes()
    print(f"Wedge graph size: {wedge_size} nodes, {wedge_G.number_of_edges()} edges")
    
    if wedge_size > 100000:
        print(f"Using fast processing algorithm for very large graphs")
        min_vc = fast_minimum_vertex_cover(wedge_G)
    elif wedge_size > 10000:
        print(f"Using simplified algorithm for large graphs")
        min_vc = simple_minimum_vertex_cover(wedge_G)
    else:
        print(f"Using standard minimum vertex cover algorithm")
        min_vc = minimum_vertex_cover_greedy(wedge_G)
    
    [labels, G] = label_weak_edges_g(G, min_vc, v_dic)
    
    return [labels, G]


def minimum_vertex_cover_greedy(G):
    
    mvc = set()
    
    remaining_edges = set(G.edges())
    if not remaining_edges:
        return mvc
    
    node_degrees = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    
    heap, _ = build_heap(G)
    
    while remaining_edges:
        try:
            _, max_degree_node = heap.pop()
        except IndexError:
            break
            
        if node_degrees[max_degree_node] == 0:
            continue
            
        incident_edges = list(G.edges(max_degree_node))
        
        if not incident_edges:
            continue
            
        mvc.add(max_degree_node)
        
        for u, v in incident_edges:
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            elif (v, u) in remaining_edges:
                remaining_edges.remove((v, u))
                
            other_node = v if u == max_degree_node else u
            if other_node in node_degrees:
                node_degrees[other_node] = max(0, node_degrees[other_node] - 1)
                
                if heap.contains(other_node):
                    heap.update(other_node, -1 * node_degrees[other_node])
    
    return mvc


def minimum_vertex_cover_greedy_plus(wedge_G, edge_dic, v_dic, te_weak, te_add):
    
    mvc = set()
    edges_add = set()
    
    remaining_edges = set(wedge_G.edges())
    if not remaining_edges:
        return [mvc, edges_add]
    
    node_degrees = {node: len(list(wedge_G.neighbors(node))) for node in wedge_G.nodes()}
    
    heap, _ = build_heap(wedge_G)
    
    weak_edge_cache = {}
    for node in wedge_G.nodes():
        query_edge = v_dic.get(node)
        weak_edge_len = 0
        
        if query_edge:
            if query_edge in te_weak:
                weak_edge_len = len(te_weak[query_edge])
            elif (query_edge[1], query_edge[0]) in te_weak:
                weak_edge_len = len(te_weak[(query_edge[1], query_edge[0])])
        
        weak_edge_cache[node] = weak_edge_len
    
    while remaining_edges:
        try:
            _, node_index = heap.pop()
        except IndexError:
            break
            
        if node_degrees[node_index] == 0:
            continue
        
        query_edge_u = v_dic[node_index]
        weak_edge_len = weak_edge_cache[node_index]
        weak_edge_flag = 1
        
        adj = list(wedge_G.edges([node_index]))
        
        for u, v in adj:
            add_edge = []
            
            query_edge_v = v_dic[v]
            if query_edge_v[0] == query_edge_u[0]:
                wedge_edge = (query_edge_u[1], query_edge_v[1])
            else:
                wedge_edge = (query_edge_u[1], query_edge_v[0])
            
            add_edge_len = 0
            if wedge_edge in te_add:
                add_edge.append(te_add[wedge_edge])
                if len(add_edge[0]) > 0:
                    add_edge_len = len(add_edge[0])
            
            if add_edge_len > weak_edge_len:
                edges_add.add(wedge_edge)
                weak_edge_flag = 0
            
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            elif (v, u) in remaining_edges:
                remaining_edges.remove((v, u))
            
            if heap.contains(v):
                node_degrees[v] = max(0, node_degrees[v] - 1)
                heap.update(v, -1 * node_degrees[v])
        
        if weak_edge_flag == 1:
            mvc.add(node_index)
    
    return [mvc, edges_add]


def label_weak_edges_g(G, min_vc, v_dic):
    edges_to_label = []
    for e in min_vc:
        edges_to_label.append(v_dic[e])
    
    edge_attr_dict = {edge: {"label": weak} for edge in edges_to_label}
    nx.set_edge_attributes(G, edge_attr_dict)
    
    labels = nx.get_edge_attributes(G, "label")
    return [labels, G]


def label_weak_edges_plus(G, min_vc, v_dic, lam, edges_add):
    G.add_edges_from(edges_add)
    
    edge_attr_dict = {}
    
    for e in min_vc:
        k = v_dic[e]
        edge_attr_dict[k] = {"label": weak, "weight": lam * 10}
    
    for edge in edges_add:
        edge_attr_dict[edge] = {"label": weak, "weight": lam * 10}
    
    nx.set_edge_attributes(G, edge_attr_dict)
    
    labels = nx.get_edge_attributes(G, "label")
    return [labels, G]


def gb_based_alg_plus(lam, G, max_wedges=0, max_edges=0):
    
    edge_attr_dict = {edge: {"label": strong, "weight": 10} for edge in G.edges()}
    nx.set_edge_attributes(G, edge_attr_dict)
    
    [wedge_G, edge_dic, v_dic, te_weak, te_add] = create_wedge_graph_org(G, max_wedges, max_edges)
    
    wedge_size = wedge_G.number_of_nodes()
    print(f"Wedge graph size: {wedge_size} nodes, {wedge_G.number_of_edges()} edges")
    
    if wedge_size > 50000:
        print(f"Using fast processing algorithm for very large graphs")
        min_vc = fast_minimum_vertex_cover(wedge_G)
        edges_add = set()
    elif wedge_size > 10000:
        print(f"Using simplified algorithm for large graphs")
        min_vc = simple_minimum_vertex_cover(wedge_G)
        edges_add = set()
    else:
        print(f"Using standard enhanced minimum vertex cover algorithm")
        [min_vc, edges_add] = minimum_vertex_cover_greedy_plus(wedge_G, edge_dic, v_dic, te_weak, te_add)
    
    [labels, G] = label_weak_edges_plus(G, min_vc, v_dic, lam, edges_add)
    
    return [labels, G]


def gb_based_alg_multi(G, k, max_wedges=0, max_edges=0):
    
    nx.set_edge_attributes(G, 1, 'label')
    
    print("Iteration 1...")
    wedge_G, edge_dic, v_dic, te_weak, te_add = create_wedge_graph_org(G, max_wedges, max_edges)
    
    wedge_size = wedge_G.number_of_nodes()
    print(f"Wedge graph size: {wedge_size} nodes, {wedge_G.number_of_edges()} edges")
    
    if wedge_size > 50000:
        print(f"Using fast processing algorithm for very large graphs")
        min_vc = fast_minimum_vertex_cover(wedge_G)
    elif wedge_size > 10000:
        print(f"Using simplified algorithm for large graphs")
        min_vc = simple_minimum_vertex_cover(wedge_G)
    else:
        print(f"Using standard minimum vertex cover algorithm")
        min_vc = minimum_vertex_cover_greedy(wedge_G)
    
    labels, G = label_weak_edges_g(G, min_vc, v_dic)
    
    i = 1
    while i < k:
        i = i + 1
        print(f"\nIteration {i}...")
        
        weak_edges = [(u, v) for u, v in G.edges() if G[u][v]['label'] == weak]
        
        if not weak_edges:
            print("No weak edges found, iteration ends")
            break
            
        print(f"This iteration processes {len(weak_edges)} weak edges")
        
        subgraph = nx.Graph()
        subgraph.add_edges_from(weak_edges)
        
        if subgraph.number_of_edges() == 0:
            print("Subgraph has no edges, iteration ends")
            break
            
        nx.set_edge_attributes(subgraph, strong, 'label')
        
        print("Building wedge graph for subgraph...")
        wedge_G, edge_dic, v_dic, te_weak, te_add = create_wedge_graph_org(subgraph, max_wedges, max_edges)
        
        if wedge_G.number_of_nodes() == 0:
            print("No wedge structure, setting all edges as strong edges")
            for u, v in subgraph.edges():
                G[u][v]["label"] = strong * i
            break
        
        wedge_size = wedge_G.number_of_nodes()
        print(f"Subgraph wedge graph size: {wedge_size} nodes, {wedge_G.number_of_edges()} edges")
        
        if wedge_size > 50000:
            print(f"Using fast processing algorithm for very large graphs")
            min_vc = fast_minimum_vertex_cover(wedge_G)
        elif wedge_size > 10000:
            print(f"Using simplified algorithm for large graphs")
            min_vc = simple_minimum_vertex_cover(wedge_G)
        else:
            print(f"Using standard minimum vertex cover algorithm")
            min_vc = minimum_vertex_cover_greedy(wedge_G)
        
        print("Marking edges...")
        edges_to_update = {}
        for e in min_vc:
            t = v_dic[e]
            edges_to_update[t] = weak
        nx.set_edge_attributes(subgraph, edges_to_update, 'label')
        
        edge_count = 0
        for u, v in subgraph.edges():
            if subgraph[u][v]['label'] == strong:
                G[u][v]["label"] = strong * i
                edge_count += 1
        
        print(f"This iteration marked {edge_count} strong edges")
    
    labels = nx.get_edge_attributes(G, "label")
    return labels, G


def save_edges_with_labels(labels, G, filename="./data/dblp/dblp-1.90.edges_labels.txt"):
    with open(filename, 'w') as file:
        for u, v in G.edges():
            label = labels.get((u, v), 'No Label')
            file.write(f"{u} {v} {label}\n")


if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Strong Triadic Closure (STC) Processing Algorithm')
    parser.add_argument('--dataset', type=str, default="lj", help='Dataset name')
    parser.add_argument('--algorithm', type=str, default="base", choices=["base", "plus", "multi"], help='Algorithm type to use: base, plus, or multi')
    parser.add_argument('--lambda', type=float, dest="lambda_val", default=0.8, help='Lambda parameter value')
    parser.add_argument('--k', type=int, default=3, help='Number of labels for multi-label algorithm')
    parser.add_argument('--parallel', action='store_true', help='Whether to enable parallel processing')
    parser.add_argument('--max-wedges', type=int, default=0, help='Maximum number of wedges to process, 0 means no limit')
    parser.add_argument('--max-edges', type=int, default=0, help='Maximum number of edges to process, 0 means no limit')
    parser.add_argument('--sample-ratio', type=float, default=1, help='Node sampling ratio, between 0-1')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"Loading dataset: {args.dataset}")
    nx_graph = load_data_STC(args.dataset)
    print(f"Dataset loaded. Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")
    
    if args.sample_ratio < 1.0:
        print(f"Performing node sampling, ratio: {args.sample_ratio}")
        nodes = list(nx_graph.nodes())
        import random
        random.seed(42)
        sample_size = int(len(nodes) * args.sample_ratio)
        sampled_nodes = random.sample(nodes, sample_size)
        nx_graph = nx.subgraph(nx_graph, sampled_nodes).copy()
        print(f"Graph size after sampling: Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")
    
    print(f"Analyzing graph structure features...")
    avg_degree = 2 * nx_graph.number_of_edges() / nx_graph.number_of_nodes()
    print(f"Average node degree: {avg_degree:.2f}")
    
    if args.algorithm == "base":
        print(f"Executing basic STC algorithm, lambda = {args.lambda_val}")
        [labels, G] = gb_based_alg(args.lambda_val, nx_graph, True, args.max_wedges, args.max_edges)
    elif args.algorithm == "plus":
        print(f"Executing enhanced STC algorithm, lambda = {args.lambda_val}")
        [labels, G] = gb_based_alg_plus(args.lambda_val, nx_graph, args.max_wedges, args.max_edges)
    elif args.algorithm == "multi":
        print(f"Executing multi-label STC algorithm, k = {args.k}")
        [labels, G] = gb_based_alg_multi(nx_graph, args.k, args.max_wedges, args.max_edges)
    
    label_counts = {}
    for _, label in labels.items():
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    print("Label distribution statistics:")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} edges")
    
    sample_suffix = f"_sample{args.sample_ratio}" if args.sample_ratio < 1.0 else ""
    output_file = f"./data/{args.dataset}/{args.dataset}-1.90.edges_labels_{args.algorithm}{sample_suffix}.txt"
    save_edges_with_labels(labels, G, output_file)
    print(f"Results saved to: {output_file}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
