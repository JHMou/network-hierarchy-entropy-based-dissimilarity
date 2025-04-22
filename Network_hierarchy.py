#this file contains all possible functions during the application of network hierarchy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import netrd


from collections import defaultdict
def compute_edge_bfs_distribution(graph):
    edge_bfs_distributions = {}
    N = graph.number_of_nodes()
    for edge in graph.edges:
        i, j = edge
        G, name = merge(graph, i, j)
        bfs_levels = defaultdict(int)
        path_lengths = nx.single_source_shortest_path_length(G, name)
        for level in path_lengths.values():
            bfs_levels[level] += 1
        edge_bfs_distributions[edge] = [count / (N - 1) for count in bfs_levels.values()]
    return edge_bfs_distributions

def compute_node_bfs_distribution(graph):
    node_bfs_distributions = {}
    N = graph.number_of_nodes()
    for node in graph.nodes():
        bfs_levels = defaultdict(int)
        path_lengths = nx.single_source_shortest_path_length(graph, node)
        for level in path_lengths.values():
            bfs_levels[level] += 1
        node_bfs_distributions[node] = [count / N for count in bfs_levels.values()]
    return node_bfs_distributions


def node_edge_measure_Cal(G):
    import math
    node_bfs_list=compute_node_bfs_distribution(G)
    edge_bfs_list = compute_edge_bfs_distribution(G)
    node_measure={}
    edge_measure={}
    for key, bfs_value in edge_bfs_list.items():
        measure=0
        for index, value in enumerate(bfs_value):
            if index<100:
                factorial = math.factorial(index)
            else:
                factorial = math.factorial(100)
            measure += value / factorial
        edge_measure[key] = measure
    for key, bfs_value in node_bfs_list.items():
        measure=0
        for index, value in enumerate(bfs_value):
            if index < 100:
                factorial = math.factorial(index)
            else:
                factorial = math.factorial(100)
            measure += value / factorial
        node_measure[key] = measure
    return node_measure,edge_measure,node_bfs_list,edge_bfs_list
def NND_GAP_cross3(G1):
    node_measure1,edge_measure1,_,_=node_edge_measure_Cal(G1)
    NND_node=0
    NND_node_per_list=[]
    for inode1 in list(G1.nodes()):
        jnode=G1[inode1]
        if len(jnode)!=0:
            edge_measure_total=0
            for jnode1 in jnode:
                edge_tupel=(inode1,jnode1)
                if edge_tupel not in list(G1.edges()):
                    edge_tupel=(jnode1,inode1)
                edge_measure_total+=edge_measure1[edge_tupel]
            edge_measure_total/=len(jnode)
            NND_node -= node_measure1[inode1] * np.log(edge_measure_total)
            NND_node_per_list.append(-node_measure1[inode1] * np.log(edge_measure_total))
    isolated_count = len([node for node, degree in G1.degree() if degree == 0])
    NND_node += isolated_count * 1
    NND_node /= len(list(G1.nodes()))
    NND_edge=0
    for iedge in list(G1.edges()):
        inode = iedge[0]
        jnode=iedge[1]
        aver=(node_measure1[inode]+node_measure1[jnode])/2
        NND_edge-= edge_measure1[iedge]* np.log(aver)
    isolated_count = len([node for node, degree in G1.degree() if degree == 0])
    NND_edge += isolated_count * 1
    NND_edge/=(len(list(G1.edges()))+isolated_count)
    return NND_node,NND_edge

def NND_GAP_node_edge(G1):
    node_measure1,edge_measure1,_,_=node_edge_measure_Cal(G1)
    node_dict={}
    for inode1 in list(G1.nodes()):
        jnode=G1[inode1]
        if len(jnode)!=0:
            edge_measure_total=0
            for jnode1 in jnode:
                edge_tupel=(inode1,jnode1)
                if edge_tupel not in list(G1.edges()):
                    edge_tupel=(jnode1,inode1)
                edge_measure_total+=edge_measure1[edge_tupel]
            edge_measure_total/=len(jnode)
            NND_node = -node_measure1[inode1] * np.log(edge_measure_total)
            node_dict[inode1]=NND_node

    edge_dict = {}
    for iedge in list(G1.edges()):
        inode = iedge[0]
        jnode=iedge[1]
        aver=(node_measure1[inode]+node_measure1[jnode])/2
        NND_edge= -edge_measure1[iedge]* np.log(aver)
        edge_dict[iedge]=NND_edge
    return node_dict,edge_dict


def similarity_2g_various_size(G1,G2):
    #deltacon,IM,NetSimile,D-meaaure,POR,
    sim_dict = {}
    #IM
    dist_obj = netrd.distance.IpsenMikhailov()
    sim_dict['IM'] = dist_obj.dist(G1, G2)
    #POR
    dist_obj = netrd.distance.PortraitDivergence()
    sim_dict['POR'] = dist_obj.dist(G1, G2)
    #NetSimile
    dist_obj = netrd.distance.NetSimile()
    sim_dict['NetSimile'] = dist_obj.dist(G1, G2)
    # NetLSD
    dist_obj = netrd.distance.NetLSD()
    sim_dict['NetLSD'] = dist_obj.dist(G1, G2)
    #D-measure
    dist_obj = netrd.distance.DMeasure()
    sim_dict['d-measure'] = dist_obj.dist(G1.subgraph(max(nx.connected_components(G1), key=len)),
                                   G2.subgraph(max(nx.connected_components(G2), key=len)))
    return sim_dict

def NND_GAP_cross3_node_edge(G1):
    node_measure1,edge_measure1,_,_=node_edge_measure_Cal(G1)
    NND_node=0
    NND_node_per_dict={}
    for inode1 in list(G1.nodes()):
        jnode=G1[inode1]
        if len(jnode)!=0:
            edge_measure_total=0
            for jnode1 in jnode:
                edge_tupel=(inode1,jnode1)
                if edge_tupel not in list(G1.edges()):
                    edge_tupel=(jnode1,inode1)
                edge_measure_total+=edge_measure1[edge_tupel]
            edge_measure_total/=len(jnode)
            NND_node -= node_measure1[inode1] * np.log(edge_measure_total)
            NND_node_per_dict[inode1]=(-node_measure1[inode1] * np.log(edge_measure_total))
    NND_edge=0
    NND_edge_per_dict = {}
    for iedge in list(G1.edges()):
        inode = iedge[0]
        jnode=iedge[1]
        aver=(node_measure1[inode]+node_measure1[jnode])/2
        NND_edge-= edge_measure1[iedge]* np.log(aver)
        NND_edge_per_dict[iedge]=-edge_measure1[iedge]* np.log(aver)
    return NND_node_per_dict,NND_edge_per_dict

import math
def calculate_bridgeness(G):
    bridgeness = {}
    cliques = list(nx.find_cliques(G))
    for edge in G.edges():
        node_x, node_y = edge
        Sx = max((len(clique) for clique in cliques if node_x in clique), default=0)
        Sy = max((len(clique) for clique in cliques if node_y in clique), default=0)
        Se = max((len(clique) for clique in cliques if set(edge).issubset(clique)), default=0)
        bridgeness[edge] = math.sqrt(Sx * Sy / Se) if Se != 0 else 0
    return bridgeness

def calculate_CN(G):
    common_neighbors = {}
    edges=list(G.edges())
    for (u,v) in edges:
        common_neighbors[(u, v)] =len(list(nx.common_neighbors(G, u, v)))
    CN_max=np.max(list(common_neighbors.values()))
    common_neighbors1 = {}
    for (u, v) in edges:
        common_neighbors1[(u, v)] = 1-common_neighbors[(u, v)]/(CN_max+1)
    return common_neighbors1

def calculate_dp(G):
    dp={}
    N=len(G.nodes())
    for (u,v) in G.edges:
        dp[(u,v)] = G.degree(u) * G.degree(v)/((N-1)**2)
    return dp

def calculate_overlap_edge(G1):
    overlap={}
    for edge in G1.edges():
        i, j = edge
        cij=len(list(nx.common_neighbors(G1,i,j)))
        ki=G1.degree(i)
        kj = G1.degree(i)
        if ki+kj-2-cij==0:
            overlap[(i, j)] = cij / (ki + kj - 2 - cij+1)
        else:
            overlap[(i,j)]=cij/(ki+kj-2-cij)
    return overlap

def calculate_diffusion_importance(G):
    diffusion_importance = {}
    for edge in G.edges():
        x, y = edge
        neighbors_x = set(G.neighbors(x))
        neighbors_y = set(G.neighbors(y))
        n_x_to_y = len(neighbors_y - neighbors_x - {x})
        n_y_to_x = len(neighbors_x - neighbors_y - {y})
        diffusion_E = (n_x_to_y + n_y_to_x) / 2
        diffusion_importance[edge] = diffusion_E
    return diffusion_importance

def BFS_Node(graph, seed):
    network_size = len(graph.nodes)
    level = [0] * network_size
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
def BFS_Node_dict(graph, seed):
    level = {}
    queue = []
    queue.append(seed)
    seen = set()
    seen.add(seed)
    level[seed]=0
    level_per=[-1]*len(list(graph.nodes()))
    while (len(queue) > 0):
        vertex = queue.pop(0)
        nodes = graph[vertex]
        for nbr in nodes:
            if nbr not in seen:
                queue.append(nbr)
                seen.add(nbr)
                level[nbr] = level[vertex] + 1
    for ite,val in level.items():
        level_per[ite]=val
    return level_per
def single_node_BFS(G, seed):
    level_single = BFS_Node_dict(G, seed)
    level_single2 = [x for x in level_single if x != -1]
    level_single_nonoverlap = list(set(level_single2))
    level_single_nonoverlap.sort()
    level_single1=np.array(level_single)
    level_set=[]
    for i in list(set(level_single2)):
        level_list=np.where(level_single1==i)
        level_set.append(list(level_list[0]))
    y_bfs = [level_single.count(x) for x in level_single_nonoverlap if x!=-1]
    return level_set,level_single, y_bfs
def BFS_distribution_G(G):
    node_list = list(G.nodes())
    single_bfs_list = []
    single_bfs_num = []
    for inode in range(len(node_list)):
        _,single_bfs, bfs_num = single_node_BFS(G, inode)
        single_bfs_num.append(bfs_num)
        single_bfs_list.append(single_bfs)
    BFS_location_dic = dict(zip(node_list, single_bfs_list))
    max_len = max(len(xx) for xx in single_bfs_num)
    M = np.array([np.concatenate([xx, np.zeros(max_len - len(xx))]) for xx in single_bfs_num])
    return M.tolist(), BFS_location_dic
def read_txt_new(filename):
    G = nx.Graph()
    edgelist=[]
    with open(filename) as file:
        for line in file:
            head, tail = [int(x) for x in line.split()]
            edgelist.append([head-1, tail-1])

    nodeset = sorted(set(itertools.chain(*edgelist)))
    G.add_nodes_from(nodeset)
    G.add_edges_from(edgelist)
    return G
def real_networks(files):
    if files.find('.csv') != -1:
        Networks = pd.read_csv(files, sep=',', encoding='gbk')
        G_edges = Networks.iloc[:, [0, 1]]
        G_edges.columns = ['source', 'target']
        G_edges_data = np.array(G_edges)
        G_edges_list = G_edges_data.tolist()
        G = nx.Graph()
        nodeset = sorted(set(itertools.chain(*G_edges_list)))
        g_size = nodeset[-1]
        G.add_nodes_from(nodeset)
        G.add_edges_from(G_edges_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest)
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
        G = G1
    if files.find('.gml') != -1:
        G = nx.read_gml(files, label='id')
    if files.find('.txt') != -1:
        G = read_txt_new(files)
    return G

def Desargues():
    G = nx.Graph()
    graphedges=[(0,1),(0,9),(0,19),(1,2),(1,12),(2,3),(2,7),(3,4),(3,18),(4,5),(4,13),(5,6),(5,16),(6,7),(6,11),(7,8),(8,9),(8,17),(9,10),(10,11),(10,15),(11,12),(12,13),(13,14),(14,15),(14,19),(15,16),(16,17),(17,18),(18,19)]
    for i in graphedges:
        G.add_edge(i[0], i[1])
    return G
def Dodecahedron():
    G = nx.Graph()
    graphedges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,0),(0,13),(1,5),(2,12),(3,10),(4,8),(6,19),(7,17),(9,16),(11,15),(14,18)]
    for i in graphedges:
        G.add_edge(i[0], i[1])
    return G












