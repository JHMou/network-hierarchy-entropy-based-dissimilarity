# -*- coding: utf-8 -*-
import copy
import networkx as nx
from random import choice
import random
import pandas as pd
import scipy.stats
import itertools
import matplotlib.pyplot as plt
import numpy as np


def BFS_Node(graph, seed):
    network_size = len(graph.nodes)
    level = [-1] * network_size
    level[seed]=0
    queue = []
    queue.append(seed)
    seen = set()
    seen.add(seed)
    while (len(queue) > 0):
        vertex = queue.pop(0)
        nodes = graph[vertex]
        for nbr in nodes:
            if nbr not in seen:
                queue.append(nbr)
                seen.add(nbr)
                level[nbr] = level[vertex] + 1
    return level


def single_node_BFS(G, seed):
    level_single = BFS_Node(G, seed)
    level_single_nonoverlap = list(set(level_single))
    if -1 in level_single_nonoverlap:
        level_single_nonoverlap.remove(-1)
    level_single_nonoverlap.sort()
    y_bfs = [level_single.count(x) for x in level_single_nonoverlap]
    return level_single, y_bfs


def BFS_distribution_G(G):
    node_list = sorted(list(G.nodes()))
    single_bfs_list = []
    single_bfs_num = []
    for inode in node_list:
        single_bfs, bfs_num = single_node_BFS(G, inode)
        single_bfs_num.append(bfs_num)
        single_bfs_list.append(single_bfs)
    BFS_location_dic = dict(zip(node_list, single_bfs_list))
    max_len = max(len(xx) for xx in single_bfs_num)
    M = np.array([np.concatenate([xx, np.zeros(max_len - len(xx))]) for xx in single_bfs_num])
    return M.tolist(), BFS_location_dic

def real_networks(files):
    if files.find('.csv') != -1:
        Networks = pd.read_csv(files, sep=',', encoding='gbk')  # encoding='utf-8'###########
        G_edges = Networks.iloc[:, [0, 1]]
        G_edges.columns = ['source', 'target']
        G_edges_data = np.array(G_edges)  # np.ndarray()
        G_edges_list = G_edges_data.tolist()  # list
        G = nx.Graph()
        nodeset = sorted(set(itertools.chain(*G_edges_list)))
        g_size = nodeset[-1]
        G.add_nodes_from(nodeset)
        G.add_edges_from(G_edges_list)
        G.remove_edges_from(nx.selfloop_edges(G))  # 去掉自环
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest)  # 获取最大连通子图
        # 重新编号
        node_list1 = []
        G1 = nx.Graph()
        edge = list(G.edges())
        for i in range(len(edge)):
            node_list1.append(edge[i][0])
            node_list1.append(edge[i][1])
        node_list1.sort()
        node_list = list(set(node_list1))
        my_dict = {}
        edgelist = []
        for index, item in enumerate(node_list):
            my_dict[item] = index
        for j in range(len(edge)):
            edgelist.append([my_dict[edge[j][0]], my_dict[edge[j][1]]])
        nodeset = sorted(set(itertools.chain(*edgelist)))
        G1.add_nodes_from(nodeset)
        G1.add_edges_from(edgelist)
        # 生成G
        G = G1
    if files.find('.gml') != -1:
        G = nx.read_gml(files, label='id')
    if files.find('.txt') != -1:
        G = read_txt_new(files)
    return G

def read_txt_new(filename):
    G = nx.Graph()
    edgelist=[]
    with open(filename) as file:
        for line in file:
            head, tail = [int(x) for x in line.split()]#新数据集初始节点编号为1
            edgelist.append([head-1, tail-1])

    nodeset = sorted(set(itertools.chain(*edgelist)))
    G.add_nodes_from(nodeset)
    G.add_edges_from(edgelist)
    return G


def multi_shortest_path_diff(G2):
    if networkx.is_connected(G2):
        path_count_distribution2 = Counter()
        for source in G2.nodes:
            for target in G2.nodes:
                if source != target:
                    # 获取所有最短路径
                    paths = list(nx.all_shortest_paths(G2, source=source, target=target))
                    # 统计最短路径的数量
                    path_count_distribution2[len(paths)] += 1
        sorted_by_key2 = dict(sorted(path_count_distribution2.items()))
        return sorted_by_key2
    else:
        connected_components = list(nx.connected_components(G2))
        path_count_distribution2 = Counter()
        for iconnected in connected_components:
            component_graph = G2.subgraph(iconnected).copy()
            for source in component_graph.nodes:
                for target in component_graph.nodes:
                    if source != target:
                        paths = list(nx.all_shortest_paths(component_graph, source=source, target=target))
                        path_count_distribution2[len(paths)] += 1
        sorted_by_key2 = dict(sorted(path_count_distribution2.items()))
        return sorted_by_key2

from dotmotif import Motif, GrandIsoExecutor
def motif(G):
    E = GrandIsoExecutor(graph=G)
    motif1 = Motif("""
    A -> B
    B -> C
    """,ignore_direction=True,exclude_automorphisms=True)
    results1 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        B -> C
        C -> A
        """, ignore_direction=True, exclude_automorphisms=True)
    results2 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        B -> C
        B -> D
        """, ignore_direction=True, exclude_automorphisms=True)
    results3 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        B -> C
        C -> D
        """, ignore_direction=True, exclude_automorphisms=True)
    results4 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        A -> C
        B -> C
        C -> D
        """, ignore_direction=True, exclude_automorphisms=True)
    results5 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        B -> C
        C -> D
        D -> A
        """, ignore_direction=True, exclude_automorphisms=True)
    results6 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        A -> C
        B -> C
        C -> D
        D -> A        
        """, ignore_direction=True, exclude_automorphisms=True)
    results7 = E.find(motif1)

    motif1 = Motif("""
        A -> B
        A -> C
        B -> C
        B -> D
        C -> D
        D -> A 
        """, ignore_direction=True, exclude_automorphisms=True)
    results8 = E.find(motif1)

    return [len(results1),len(results2),len(results3),len(results4),len(results5),len(results6),len(results7),len(results8)]













