# coding: utf-8
# author: Silvan
# file  : STC.py
# time  : 2024-12-29 10:31
import time
from collections import defaultdict
import networkx as nx
from matplotlib import pyplot as plt

from Heap import build_heap

# label for edges : constants   原始版本复制过来的版本
weak = 0
strong = 1

def draw_graph(G):
    # 获取节点的位置，这里使用 spring_layout 来模拟节点的物理布局
    pos = nx.spring_layout(G, seed=1)
    # 绘制节点
    # nx.draw_networkx_nodes(nx_graph, pos, node_size=40)
    # 绘制边
    # nx.draw_networkx_edges(nx_graph, pos, alpha=0.1)
    nx.draw_networkx(G, pos, node_color='green', edge_color='red', node_size=500, width=2)
    # 绘制图形
    # plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图形
    # 使用 spring_layout 设置节点位置，并设置随机种子以获得可重复的结果
    # pos = nx.spring_layout(G, seed=1)
    #
    # # 创建一个足够大的图形窗口
    # plt.figure(figsize=(14, 8))
    #
    # # 绘制图形，调整节点大小和边的宽度
    # nx.draw_networkx(G, pos, node_color='green', edge_color='red', node_size=500, width=2)
    #
    # # 保存图形为 SVG 文件
    # # plt.savefig('karate_club_graph.svg')
    #
    # # 显示图形
    # plt.show()

def load_data_STC(dataset_name):
    edge_labels = open(f"./data/{dataset_name}/{dataset_name}-1.90.ungraph.txt")
    labels = [[int(i) for i in l.split()] for l in edge_labels]  #edge_labels 17893
    edges = [edge[:2] for edge in labels]

    # Remove self-loop edges
    # edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}  #amazon 6926
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    nx_graph = nx.Graph(edges)
    nx_graph.add_nodes_from(nodes)

    # draw_graph(nx_graph)
    return nx_graph


def create_wedge_graph_org(G):
    """
    接受一个图 G 作为输入，
    并创建一个所谓的“楔形图”（wedge graph），这是一个特殊类型的图，用于在社交网络分析中研究强三元闭包（STC）属性。
    这个函数的目的是将一个图转换为一个楔形图，这个楔形图可以用来分析和理解原始图中的强三元闭包属性。
    在社交网络分析中，这有助于识别网络中的紧密连接的群体。
    """
    nodes = list(G.nodes)  # 获取输入图 G 的所有节点

    # adjacency vertex dictionary
    adj_dic = {}  # 是一个字典，用于存储每个节点的邻居节点集合
    for v in nodes:
        neighbour_nodes = [n for n in G.neighbors(v)]
        adj_dic[v] = set(neighbour_nodes)

    # adjacency edge dictionary
    adj_edge_dic = {}  # 用于存储每个节点的邻居边集合
    for v in nodes:
        neighbour_edges = G.edges(v)
        adj_edge_dic[v] = set(neighbour_edges)

    # key: root of wedge, values: list(two neighbors) (i,(j,k))
    wedge = defaultdict(set)  # wedge是一个使用set作为值的字典，用于存储楔形结构。键是楔形的根节点，值是一对邻居节点的集合。

    # set of edges which contribute to wedges
    edge_set = set()  # edge_set 是一个集合，用于存储构成楔形的边
    # 代码遍历每个节点i及其邻居j，检查是否存在一个楔形结构，即节点i与j和k都有连接，但j和k之间没有直接连接。
    for i in nodes:
        ne = adj_dic[i]

        for j in ne:  # ne是当前节点i所有的邻居集合
            ne2 = adj_dic[j]

            for k in ne2:  # ne2是邻居的邻居集合
                if i == j or i == k or j == k:
                    continue
                if k in ne:
                    continue
                # 如果找到楔形结构，它将 j 作为根节点，并将 (i, k) 对添加到 wedge 字典中。
                if i in ne2 and k in ne2:
                    wedge[j].add((i, k))

                    # undirected  将构成楔形的边 (i, j) 和 (j, k) 添加到 edge_set 中
                    if i > j:
                        edge_set.add((j, i))  # 两个节点标识一条边，形成边集合，并值小的节点在前，值大的节点在后
                    else:
                        edge_set.add((i, j))

                    if j > k:
                        edge_set.add((k, j))
                    else:
                        edge_set.add((j, k))

    # assign each edge an id (number edges to be used as vertices in wedge graph)
    # 为 edge_set 中的每条边分配一个唯一的ID，这些ID将用作楔形图中的顶点
    values = range(0, len(edge_set))  # 生成序列
    keys = list(edge_set)  # 将set转为list

    # nodes in wedge graph
    # edge_dic 是一个字典，将边映射到它们的ID
    edge_dic = dict(zip(keys, values))

    #  id, edge
    #  v_dic 是 edge_dic 的反向映射，将ID映射回边
    v_dic = dict(zip(values, keys))

    # edges in wedge graph
    # wedge_edges 是一个集合，用于存储楔形图中的边
    wedge_edges = set()
    # 初始化一个空字典来存储triangle_edges边和对应的值
    te_weak = {}
    te_add = {}

    # construct edges in wedge graph
    # 代码遍历 wedge 字典，为每对邻居节点创建楔形图中的边。
    # 这通过查找每对节点在edge_dic中的ID并添加到 wedge_edges 集合中来完成
    for i, (k, v) in enumerate(wedge.items()):
        lst = v

        for (a, b) in lst:
            # if (a, k) not in te_weak and (k, a) not in te_weak:
            #     te_weak[(k, a)] = []
            # te_weak[(a, k)].append((k, b))
            # te_weak[(k, a)].append((k, b))
            #
            # if (b, k) not in te_weak and (k, b) not in te_weak:
            #     te_weak[(k, b)] = []
            # te_weak[(b, k)].append((k, a))
            # te_weak[(k, b)].append((k, a))
            if (a, k) in te_weak:
                te_weak[(a, k)].append((k, b))
            elif (k, a) in te_weak:
                te_weak[(k, a)].append((k, b))
            else:
                te_weak[(k, a)] = []
                te_weak[(k, a)].append((k, b))

            # if (b, k) in te_weak:
            #     te_weak[(b, k)].append((k, a))
            # elif (k, b) in te_weak:
            #     te_weak[(k, b)].append((k, a))
            # else:
            #     te_weak[(k, b)] = []
            #     te_weak[(k, b)].append((k, a))

            if (a, b) not in te_add:
                te_add[(a, b)] = []
            te_add[(a, b)].append([(k, b), (k, a)])

            if (k, a) in edge_dic.keys():
                e1 = edge_dic[(k, a)]
            else:
                e1 = edge_dic[(a, k)]

            if (k, b) in edge_dic.keys():
                e2 = edge_dic[(k, b)]
            else:
                e2 = edge_dic[(b, k)]

            wedge_edges.add((e1, e2))
    # 使用 wedge_edges 集合，代码创建了一个新的图 wedge_G，这是一个无向图，代表了原始图 G 的楔形结构
    wedge_G = nx.Graph()
    wedge_G.add_edges_from(wedge_edges)
    wedge_G = wedge_G.to_undirected()
    # draw_graph(wedge_G)

    # remove isolated nodes 代码移除了 wedge_G 中的孤立节点，并打印出楔形图的节点数和边数
    wedge_G.remove_nodes_from(list(nx.isolates(wedge_G)))

    print('V(W): {}  E(W): {}'.format(wedge_G.number_of_nodes(), wedge_G.number_of_edges()))
    # 函数返回一个列表，包含楔形图 wedge_G，边到ID的映射 edge_dic，以及ID到边的映射 v_dic
    return [wedge_G, edge_dic, v_dic, te_weak, te_add]


def minimum_vertex_cover_greedy_plus(wedge_G, edge_dic, v_dic, te_weak, te_add):
    """使用greedy_plus算法来为图 G 找到一个最小顶点覆盖的近似解"""
    mvc = set()  # 初始化一个空集合 mvc 来存储最小顶点覆盖中的顶点
    edges = set(wedge_G.edges)
    # 返回一个优先队列（或堆）和每个顶点的度数。优先队列用于存储顶点，按照它们的度数降序排列。
    heap, degrees = build_heap(wedge_G)
    edges_add = set()  # 要添加的边

    while len(edges) > 0:
        # remove node with max degree
        _, node_index = heap.pop()  # 从优先队列 heap 中弹出度数最高的顶点 node_index
        adj = set(wedge_G.edges([node_index]))  # 获取顶点 node_index 的邻居边集合 adj
        weak_edge_len = 0
        weak_edge_flag = 1
        query_edge_u = v_dic[node_index]
        if query_edge_u in te_weak:
            weak_edge = [te_weak[query_edge_u]]
            weak_edge_len = len(weak_edge[0])
        elif (query_edge_u[1], query_edge_u[0]) in te_weak:
            weak_edge = [te_weak[(query_edge_u[1], query_edge_u[0])]]
            weak_edge_len = len(weak_edge[0])
        # 遍历顶点 node_index 的所有邻居边 (u, v)，并从 edges 集合中移除这些边，因为它们已经被顶点 node_index 覆盖
        # 如果对应的边能覆盖更多的三角形，添加该边，并移除相关的边
        # 如果覆盖的三角形小于当前节点能覆盖的三角形，移除这条边
        for u, v in adj:
            add_edge = []

            query_edge_v = v_dic[v]
            if query_edge_v[0] == query_edge_u[0]:
                wedge_edge = (query_edge_u[1], query_edge_v[1])
            else:
                wedge_edge = (query_edge_u[1], query_edge_v[0])
            # 键是楔形边，根据边查询有多少楔形三角形
            if wedge_edge in te_add:
                add_edge.append(te_add[wedge_edge])

            add_edge_len = 0
            if add_edge and len(add_edge) > 0:
                add_edge_len = len(add_edge[0])
            # degree_u = degree_u +1

            if add_edge_len > weak_edge_len:
                edges_add.add(wedge_edge)
                weak_edge_flag = 0
            # remove edge from list
            edges.discard((u, v))
            edges.discard((v, u))
            # 对于每个邻居顶点 v，如果它还在优先队列中，更新它的度数 new_degree 并调整优先队列。
            # 这里 degrees[v] 存储顶点 v 的当前度数，heap.update(v, -1 * new_degree) 用于更新优先队列中顶点 v 的度数。
            # update neighbors
            if heap.contains(v):
                new_degree = degrees[v] - 1
                # update index
                degrees[v] = new_degree
                # update heap
                heap.update(v, -1 * new_degree)

        # add node in mvc 将顶点 node_index 添加到 mvc 集合中，表示这个顶点被选为最小顶点覆盖的一部分
        if weak_edge_flag == 1:
            mvc.add(node_index)

    return [mvc, edges_add]  # 返回包含最小顶点覆盖顶点集合 mvc


def minimum_vertex_cover_greedy(G):
    """
    使用贪婪算法来为图 G 找到一个最小顶点覆盖的近似解
    :param G:
    :return:
    """
    mvc = set()  # 初始化一个空集合 mvc 来存储最小顶点覆盖中的顶点

    edges = set(G.edges)
    # heap 和 degrees 是通过调用 build_heap(G) 函数构建的，
    # 这个函数可能返回一个优先队列（或堆）和每个顶点的度数。优先队列用于存储顶点，按照它们的度数降序排列。
    heap, degrees = build_heap(G)

    while len(edges) > 0:
        # remove node with max degree
        _, node_index = heap.pop()  # 从优先队列 heap 中弹出度数最高的顶点 node_index
        adj = set(G.edges([node_index]))  # 获取顶点 node_index 的邻居边集合 adj
        # 遍历顶点 node_index 的所有邻居边 (u, v)，并从 edges 集合中移除这些边，因为它们已经被顶点 node_index 覆盖
        for u, v in adj:
            # remove edge from list
            edges.discard((u, v))
            edges.discard((v, u))
            # 对于每个邻居顶点 v，如果它还在优先队列中，更新它的度数 new_degree 并调整优先队列。
            # 这里 degrees[v] 存储顶点 v 的当前度数，heap.update(v, -1 * new_degree) 用于更新优先队列中顶点 v 的度数。
            # update neighbors
            if heap.contains(v):
                new_degree = degrees[v] - 1
                # update index
                degrees[v] = new_degree
                # update heap
                heap.update(v, -1 * new_degree)

        # add node in mvc 将顶点 node_index 添加到 mvc 集合中，表示这个顶点被选为最小顶点覆盖的一部分
        mvc.add(node_index)

    return mvc  # 返回包含最小顶点覆盖顶点集合 mvc


def label_weak_edges_g(G, min_vc, v_dic):
    for e in min_vc:
        k = v_dic[e]
        G.edges[k]["label"] = weak

    labels = nx.get_edge_attributes(G, "label")

    return [labels, G]


def label_weak_edges_plus(G, min_vc, v_dic, lam, edges_add):
    G.add_edges_from(edges_add)
    for e in min_vc:
        k = v_dic[e]
        G.edges[k]["label"] = weak
        G.edges[k]["weight"] = lam * 10  # 弱边的权重设置为 lam*10
    for i in edges_add:
        G.edges[i]["label"] = weak
        G.edges[i]["weight"] = lam * 10  # 弱边的权重设置为 lam*10
    labels = nx.get_edge_attributes(G, "label")
    return [labels, G]


def gb_based_alg(lam, G, isGdy=True):
    """通过楔形图和最小顶点覆盖来分析社交网络中的边的强度，并找到网络中的密集子图。
    这个过程涉及到图的复制、子图的创建、边的属性设置以及基于特定算法的边的标记。
    最后，代码计算了标记后的子图中强边和弱边的数量，这可以用于进一步的社交网络分析。
    :param lam: 权重参数
    :param G: 输入图
    :param isGdy: 一个布尔值，用于选择不同的最小顶点覆盖算法
    """
    for (a, b) in G.edges():  # 首先遍历图 G 的所有边，并将每条边的属性设置为强（"strong"）和权重 10
        nx.set_edge_attributes(G, {(a, b): {"label": strong, "weight": 10}})

    # create wedge graph 创建楔形图 wedge_G，以及边到ID的映射 edge_dic 和 ID 到边的映射 v_dic
    [wedge_G, edge_dic, v_dic, te_weak, te_add] = create_wedge_graph_org(G)

    # run min_vertex_cover 根据 isGdy 的值，选择不同的算法来找到楔形图 wedge_G 的最小顶点覆盖 min_vc
    min_vc = minimum_vertex_cover_greedy(wedge_G)
    # find weak and strong labels 根据最小顶点覆盖 min_vc 和参数 lam 来标记图 G 中的弱边和强边
    [labels, G] = label_weak_edges_g(G, min_vc, v_dic)
    return [labels, G]


def gb_based_alg_plus(lam, G):
    for (a, b) in G.edges():  # 首先遍历图 G 的所有边，并将每条边的属性设置为强（"strong"）和权重 10
        nx.set_edge_attributes(G, {(a, b): {"label": strong, "weight": 10}})

    # create wedge graph 创建楔形图 wedge_G，以及边到ID的映射 edge_dic 和 ID 到边的映射 v_dic
    [wedge_G, edge_dic, v_dic, te_weak, te_add] = create_wedge_graph_org(G)

    # run min_vertex_cover 根据 isGdy 的值，选择不同的算法来找到楔形图 wedge_G 的最小顶点覆盖 min_vc
    [min_vc, edges_add] = minimum_vertex_cover_greedy_plus(wedge_G, edge_dic, v_dic, te_weak, te_add)
    # find weak and strong labels 根据最小顶点覆盖 min_vc 和参数 lam 来标记图 G 中的弱边和强边
    [labels, G] = label_weak_edges_plus(G, min_vc, v_dic, lam, edges_add)
    return [labels, G]


def gb_based_alg_multi(G, k):
    """多标签标注算法"""
    # nx.set_edge_attributes(G, -1, 'label')
    nx.set_edge_attributes(G, 1, 'label')
    wedge_G, edge_dic, v_dic, te_weak, te_add = create_wedge_graph_org(G)
    min_vc = minimum_vertex_cover_greedy(wedge_G)
    labels, G = label_weak_edges_g(G, min_vc, v_dic)
    # i = -1
    i = 1
    while i < k:  # 减去第一次迭代，因为已经在外部处理了
        i = i + 1
        weak_edges = [(u, v) for u, v in G.edges() if G[u][v]['label'] == weak]

        if not weak_edges:
            break  # 如果没有弱边，退出循环

        subgraph = nx.Graph()
        subgraph.add_edges_from(weak_edges)
        subgraph = subgraph.to_undirected()
        subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
        nx.set_edge_attributes(subgraph, strong, 'label')

        # 对子图进行操作
        wedge_G, edge_dic, v_dic, te_weak, te_add = create_wedge_graph_org(subgraph)
        min_vc = minimum_vertex_cover_greedy(wedge_G)
        for e in min_vc:
            t = v_dic[e]
            subgraph.edges[t]["label"] = weak

        # 更新原图 G 中的边标签
        for u, v in subgraph.edges():
            if subgraph[u][v]['label'] == strong:
                G[u][v]["label"] = strong * i

    labels = nx.get_edge_attributes(G, "label")
    return labels, G


def save_edges_with_labels(labels, G, filename="./data/facebook/facebook-1.90.edges_labels.txt"):
    # 打开文件以写入
    with open(filename, 'w') as file:
        # 遍历图G中的所有边
        for u, v in G.edges():
            # 获取边(u, v)的标签
            label = labels.get((u, v), 'No Label')  # 如果边没有标签，则默认为'No Label'
            # 将边和标签写入文件，这里以空格分隔边的两个节点和标签  weak = 0
            file.write(f"{u} {v} {label}\n")
            file.write(f"{v} {u} {label}\n")


if __name__ == "__main__":
    start_time = time.time()
    nx_graph = load_data_STC("lj")
    # edges = [[1, 2], [2, 3], [3, 4], [4, 5], [2, 5]]
    # edges = [[1, 2],[1, 3], [2, 3], [1, 4], [2, 4], [3, 5],[3, 0]]
    # nodes = {node for e in edges for node in e}
    # nx_graph = nx.Graph(edges)
    # nx_graph.add_nodes_from(nodes)

    # nx_graph = nx.karate_club_graph()  # 空手道俱乐部图
    # draw_graph(nx_graph)

    [labels, G] = gb_based_alg(0.8, nx_graph)
    # [labels, G] = gb_based_alg_plus(0.8,nx_graph)
    # [labels, G] = gb_based_alg_multi(nx_graph, 3)
    # 调用函数保存边和标签到文件
    save_edges_with_labels(labels, G)
    # 输出总运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")
