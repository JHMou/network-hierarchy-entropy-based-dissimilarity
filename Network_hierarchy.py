#this file contains all possible functions during the application of network hierarchy
import pandas as pd
import itertools
import networkx as nx
import numpy as np
from collections import defaultdict
import netrd


def merge(G, i, j):
    """Generate a merged network by contracting nodes i and j into a new supernode.

    This operation creates a new network where nodes i and j are replaced by a single new node.
    The new node inherits all connections from both original nodes (without duplicate edges).
    Parameters:
        G : networkx.Graph
            The input graph to be processed.
        i : int
            The first node connected by an edge e_ij.
        j : int
            The second node connected by an edge e_ij.
    return:
        G1 : networkx.Graph
            A new graph with nodes i and j merged into a new supernode.
        name : int
            The identifier of the newly created supernode. This is typically assigned as N + 1.
    """
    import copy
    G1 = copy.deepcopy(G)
    i_neigh = list(G1[i])
    j_neigh = list(G1[j])
    new_neigh = list(set(i_neigh + j_neigh))
    name=np.max(list(G1.nodes()))+1
    G1.add_edges_from([(p, name) for p in new_neigh])
    G1.remove_nodes_from([i, j])
    return G1, name



def compute_bfs_distributions(graph):
    """
    node hierarchy and edge hierarchy for graphs
    Parameters:
        graph : networkx.Graph
    return:
        node_bfs_distributions:
            node hierarchy for all nodes
        edge_bfs_distributions:
            edge hierarchy for all edges
    """
    N = graph.number_of_nodes()
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    all_pairs_lengths = dict(nx.all_pairs_shortest_path_length(graph))

    # the distance distribution for nodes
    node_bfs_distributions = {}
    for node in nodes:
        level_counts = defaultdict(int)
        for distance in all_pairs_lengths[node].values():
            level_counts[distance] += 1
        # normalization
        distribution = [level_counts[d] / N for d in sorted(level_counts.keys())]
        node_bfs_distributions[node] = distribution

    # the distance distribution for edges by the rule "min_dist = min(dist_to_i, dist_to_j)"
    edge_bfs_distributions = {}
    for i, j in edges:
        level_counts = defaultdict(int)
        level_counts[0] = 1  # the merged edges

        for target in nodes:
            if target == i or target == j:
                continue
            # distance to the merged edge is the minimal distance to the endpoint i and j
            dist_to_i = all_pairs_lengths[i].get(target, float('inf'))
            dist_to_j = all_pairs_lengths[j].get(target, float('inf'))
            min_dist = min(dist_to_i, dist_to_j)
            if min_dist != float('inf'):
                level_counts[min_dist] += 1
        # normalization
        distribution = [level_counts[d] / (N - 1) for d in sorted(level_counts.keys())]
        edge_bfs_distributions[(i, j)] = distribution
    return node_bfs_distributions, edge_bfs_distributions


def NND_GAP_cross3(G1):
    """
    NHE-EHE for graph
    Parameters:
        G1 : networkx.Graph
    return:
        NND_node:
            node hierarchy entropy for graph
        NND_edge:
            edge hierarchy entropy for graph
    """
    node_measure1, edge_measure1 = node_edge_measure_Cal(G1)

    nodes = G1.nodes()
    edges = G1.edges()
    num_nodes = G1.number_of_nodes()
    num_edges = G1.number_of_edges()

    # the number of isolated nodes
    isolated_count = 0
    node_contributions = 0.0
    NND_node_per_list = []

    # Caculation for node hierarchy entropy
    for node in nodes:
        neighbors = list(G1.neighbors(node))
        if not neighbors:
            isolated_count += 1
            node_contributions += 1.0
            NND_node_per_list.append(1.0)
        else:
            edge_measure_total = 0.0
            for neighbor in neighbors:
                edge_tuple = (min(node, neighbor), max(node, neighbor))
                if edge_tuple not in edge_measure1:
                    edge_tuple = (max(node, neighbor), min(node, neighbor))
                edge_measure_total += edge_measure1[edge_tuple]

            avg_edge_measure = edge_measure_total / len(neighbors)
            if avg_edge_measure > 0:
                contribution = -node_measure1[node] * np.log(avg_edge_measure)
                node_contributions += contribution
                NND_node_per_list.append(contribution)
            else:
                NND_node_per_list.append(0.0)

    NND_node = node_contributions / num_nodes

    # Caculation for edge hierarchy entropy
    edge_contributions = 0.0
    for u, v in edges:
        aver = (node_measure1[u] + node_measure1[v]) / 2
        if aver > 0:
            edge_contributions -= edge_measure1[(u, v)] * np.log(aver)


    edge_contributions += isolated_count * 1.0
    NND_edge = edge_contributions / (num_edges + isolated_count)

    return NND_node, NND_edge




def node_edge_measure_Cal(G):
    """
    NHC-EHC for graph
    Parameters:
        G : networkx.Graph
    return:
        node_measure:
            node hierarchy centrality for each node
        edge_measure:
            edge hierarchy centrality for each edge
    """
    import math
    node_bfs_list, edge_bfs_list = compute_bfs_distributions(G)
    node_measure={}
    edge_measure={}
    #edge hierarchy centrality
    for key, bfs_value in edge_bfs_list.items():
        measure=0
        for index, value in enumerate(bfs_value):
            if index<100:#ignore the layers larger than 100
                factorial = math.factorial(index)
            else:
                factorial = math.factorial(100)
            measure += value / factorial
        edge_measure[key] = measure
    #node hierarchy centrality
    for key, bfs_value in node_bfs_list.items():
        measure=0
        for index, value in enumerate(bfs_value):
            if index < 100:
                factorial = math.factorial(index)
            else:
                factorial = math.factorial(100)
            measure += value / factorial
        node_measure[key] = measure
    return node_measure,edge_measure



def convert2line_graph(g1):
    """
    line graph for g1
    Parameters:
        g1 : networkx.Graph
    return:
        new_graph1:
            networkx.Graph
    """
    gl1 = nx.line_graph(g1)
    line_graph_nodes = list(gl1.nodes())
    line_graph_edges = list(gl1.edges())
    new_graph1 = nx.Graph()
    # unique ID for each edge
    edge_id_map = {edge: idx for idx, edge in enumerate(line_graph_nodes)}
    # the edges in line graph
    for edge in line_graph_edges:
        new_graph1.add_node(edge_id_map[edge[0]])
        new_graph1.add_node(edge_id_map[edge[1]])
        if edge[0] in line_graph_nodes and edge[1] in line_graph_nodes:
            new_graph1.add_edge(edge_id_map[edge[0]], edge_id_map[edge[1]])
    return new_graph1
def NND_avg(G1):######节点和连边中心性的平均值
    """
    the average NHC and EHC for G1
    """
    node_measure1,edge_measure1=node_edge_measure_Cal(G1)
    return np.average(list(node_measure1.values())),np.average(list(edge_measure1.values()))


def similarity_2g_various_size(G1,G2):
    """
    distance between G1 and G2 by other metrics
    Parameters:
        G1 : networkx.Graph
        G2 : networkx.Graph
    return:
        sim_dict:
            the distance through IM,NetSimile,D-meaaure,POR,NetLSD
    """
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
    """
    Tij for each edge and Tk for each node
    Parameters:
        G1 : networkx.Graph
    return:
        NND_node_per_dict:
            Tk for each node
        NND_edge_per_dict:
            Tij for each edge
    """
    node_measure1,edge_measure1=node_edge_measure_Cal(G1)
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
    """
    calculate_bridgeness for each node
    Parameters:
        G : networkx.Graph
    return:
        bridgeness:
            the bridgeness for each edge
    """
    bridgeness = {}
    cliques = list(nx.find_cliques(G))
    for edge in G.edges():
        node_x, node_y = edge
        # the maximum size of cliques containing a specific node
        Sx = max((len(clique) for clique in cliques if node_x in clique), default=0)
        Sy = max((len(clique) for clique in cliques if node_y in clique), default=0)
        # the maximum size of cliques containing a specific edge
        Se = max((len(clique) for clique in cliques if set(edge).issubset(clique)), default=0)

        bridgeness[edge] = math.sqrt(Sx * Sy / Se) if Se != 0 else 0

    return bridgeness

def calculate_CN(G):
    """
     calculate common neighbors for each edge
     Parameters:
         G : networkx.Graph
     return:
         common_neighbors1:
             common neighbors for each edge
     """
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
    """
     calculate degree product for each edge
     Parameters:
         G : networkx.Graph
     return:
         dp:
             degree product for each edge
     """
    dp={}
    N=len(G.nodes())
    for (u,v) in G.edges:
        dp[(u,v)] = G.degree(u) * G.degree(v)/((N-1)**2)
    return dp



def calculate_overlap_edge(G1):
    """
     calculate topological overlap (TO) for each edge
     Parameters:
         G1 : networkx.Graph
     return:
         overlap:
             TO for each edge
     """
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
    """
     calculate diffusion importance (DI) for each edge
     Parameters:
         G1 : networkx.Graph
     return:
         diffusion_importance:
             DI for each edge
     """
    diffusion_importance = {}
    for edge in G.edges():
        x, y = edge
        # the neighbor of node x and y
        neighbors_x = set(G.neighbors(x))
        neighbors_y = set(G.neighbors(y))

        n_x_to_y = len(neighbors_y - neighbors_x - {x})
        n_y_to_x = len(neighbors_x - neighbors_y - {y})

        diffusion_E = (n_x_to_y + n_y_to_x) / 2
        diffusion_importance[edge] = diffusion_E
    return diffusion_importance


def BFS_Node_dict(graph, seed):
    """
     calculate the distance away from seed node
     Parameters:
         graph : networkx.Graph
         seed: the root node
     return:
         level_per:
             the layer for each node originating node seed
     """
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
    """
     calculate the distance away from seed node
     Parameters:
         G : networkx.Graph
         seed: the root node
     return:
         level_single:
             the layer for each node originating node seed
         level_set:
            the list of node at the same layer
         y_bfs:
            the number of nodes at each layer
     """
    level_single = BFS_Node_dict(G, seed)  # the layer for each node originating node seed

    level_single2 = [x for x in level_single if x != -1]
    level_single_nonoverlap = list(set(level_single2))
    level_single_nonoverlap.sort()
    #the list of node at the same layer
    level_single1=np.array(level_single)
    level_set=[]
    for i in list(set(level_single2)):
        level_list=np.where(level_single1==i)
        level_set.append(list(level_list[0]))
    # the number of nodes at each layer
    y_bfs = [level_single.count(x) for x in level_single_nonoverlap if x!=-1]
    return level_set,level_single, y_bfs
def BFS_distribution_G(G):
    """
     calculate the network hierarchy matrix
     Parameters:
         G : networkx.Graph
     return:
         M_list:
             the network hierarchy matrix of graph G
     """

    node_list = list(G.nodes())
    single_bfs_list = []
    single_bfs_num = []
    for inode in range(len(node_list)):
        _,single_bfs, bfs_num = single_node_BFS(G, inode)
        single_bfs_num.append(bfs_num)
        single_bfs_list.append(single_bfs)
    BFS_location_dic = dict(zip(node_list, single_bfs_list))
    # aggragate all node hierarchy vector to network hierarchy matrix
    max_len = max(len(xx) for xx in single_bfs_num)
    M = np.array([np.concatenate([xx, np.zeros(max_len - len(xx))]) for xx in single_bfs_num])
    M_list=M.tolist()
    return M_list
def read_txt_new(filename):
    """
     load network from txt file
     Parameters:
         filename : the txt file
     return:
         G : networkx.Graph
     """
    G = nx.Graph()
    edgelist=[]
    with open(filename) as file:
        for line in file:
            head, tail = [int(x) for x in line.split()]
            edgelist.append([head, tail])

    nodeset = sorted(set(itertools.chain(*edgelist)))
    G.add_nodes_from(nodeset)
    G.add_edges_from(edgelist)
    # reindex for nodes and edges
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
    return G
def real_networks(files):
    """
     load network from general file
     Parameters:
         files : the txt file
     return:
         G : networkx.Graph
     """
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
        G = G.subgraph(largest)  # the gaint connected component
        # reindex
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
    """
     generate Desargues graph
     Parameters:
         none
     return:
         G : networkx.Graph
     """
    G = nx.Graph()
    graphedges=[(0,1),(0,9),(0,19),(1,2),(1,12),(2,3),(2,7),(3,4),(3,18),(4,5),(4,13),(5,6),(5,16),(6,7),(6,11),(7,8),(8,9),(8,17),(9,10),(10,11),(10,15),(11,12),(12,13),(13,14),(14,15),(14,19),(15,16),(16,17),(17,18),(18,19)]
    for i in graphedges:
        G.add_edge(i[0], i[1])
    return G
def Dodecahedron():
    """
     generate Dodecahedron graph
     Parameters:
         none
     return:
         G : networkx.Graph
     """
    G = nx.Graph()
    graphedges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,0),(0,13),(1,5),(2,12),(3,10),(4,8),(6,19),(7,17),(9,16),(11,15),(14,18)]
    for i in graphedges:
        G.add_edge(i[0], i[1])
    return G


def JS_divergence(p,q):
    import scipy.stats
    mid=len(q)-len(p)
    if mid>0:
        for j in range(mid):
            p.append(0)
    if mid<0:
        for j in range(-mid):
            q.append(0)
    M=(np.array(p)+np.array(q))/2
    return 0.5*scipy.stats.entropy(np.array(p),M)+0.5*scipy.stats.entropy(np.array(q), M)







