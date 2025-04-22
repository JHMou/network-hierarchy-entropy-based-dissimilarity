
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import itertools
import seaborn as sns
import Network_hierarchy as nh
from tqdm import tqdm
import BFS_STATISTICS as bfs_stat
import LFR_init as lfr_init
from multiprocessing import Pool
import math

def edge_bfs(G1):
    edgealllist = G1.edges()
    edgealllist_df = pd.DataFrame(list(edgealllist))
    edgealllist_df.columns = ['source', 'target']
    edgealllist_df = edgealllist_df.sort_values(by="source",ascending=True)
    edge_bfs_list=[]
    edge_bfs_Desargues={}
    for i in range(edgealllist_df.shape[0]):
        ui=edgealllist_df.iloc[i,0]
        uj=edgealllist_df.iloc[i,1]
        BFS_num2_u=nh.merge_all_edges(G1,ui,uj)
        edge_bfs_Desargues[(ui,uj)]=list(BFS_num2_u[-1])
        edge_bfs_list.append(BFS_num2_u[-1])
    return edge_bfs_Desargues


def community_network(tau1):#检查存在社区结构的网络的二维分布情况
    if True:
        mu_list = np.arange(1, 7)
        colors = [
            (1, 0, 0),  # red
            (0, 0, 1),  # blue
            (0, 1, 0),  # green
            (1, 0.5, 0),  # orange
            (0.5, 0, 0.5),  # purple
            (0.5, 0.25, 0),  # brown
            (1, 0.75, 0.8),  # pink
        ]
        plt.figure()
        for mu in mu_list:
            node_similarity = []
            edge_similarity = []
            count = 50  # 重复20次试验
            for ic in range(count):
                paras = {"n": 100, "tau1": tau1, "tau2": 2, "mu": mu / 10, "average_d": 4, "max_d": 20,
                         "min_community": 20,
                         "max_community": 50, "seed": ic}
                args = tuple(paras.values())
                G1, true_label = lfr_init.LFR_init(*args)
                a, b = nh.NND_GAP_cross3(G1)
                node_similarity.append(a)
                edge_similarity.append(b)
            plt.scatter(node_similarity, edge_similarity, color=colors[mu - 1])
        plt.show()

import os
import scipy.io as scio
def read_networks(base_directory,first_level_folder):
    categorized_data = {}
    node_similarity = []
    edge_similarity = []
    # Walk through the second-level directory
    first_level_path = os.path.join(base_directory, first_level_folder)
    G_agg = []
    for second_level_folder in tqdm(os.listdir(first_level_path)):
        second_level_path = os.path.join(first_level_path, second_level_folder)
        # Check if it's a directory
        if os.path.isdir(second_level_path):
            # Iterate through .mat files in the second-level directory
            for file_name in os.listdir(second_level_path):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(second_level_path, file_name)
                    # Load the .mat file
                    mat_data = scio.loadmat(file_path)
                    G = nx.from_numpy_matrix((mat_data['G']))
                    G1 = G.to_undirected()
                    G1.remove_edges_from(nx.selfloop_edges(G1))
                    a, b = nh.NND_GAP_cross3(G1)
                    node_similarity.append(a)
                    edge_similarity.append(b)
                    G_agg.append([G1,(a,b)])
    categorized_data[first_level_folder]=G_agg
    return categorized_data
import pickle

def read_173networks():
    base_directory='./network-similarity/dataset/'
    categorized_data = []
    cpu_cores = 23
    pool = Pool(cpu_cores)
    results = []
    for first_level_folder in tqdm(os.listdir(base_directory)):
        results.append(pool.apply_async(read_networks, args=(base_directory, first_level_folder)))
    pool.close()
    pool.join()
    print('done')
    for result in [r.get() for r in results]:
        categorized_data.append(result)
    with open('./network-similarity/real_networks_2dimension.pkl', 'wb') as file:
        pickle.dump(categorized_data, file)
    print(len(categorized_data))




from collections import defaultdict
def read_networks_distance(base_directory,first_level_folder):
    first_level_path = os.path.join(base_directory, first_level_folder)
    G_dict={}
    for second_level_folder in tqdm(os.listdir(first_level_path)):
        second_level_path = os.path.join(first_level_path, second_level_folder)
        # Check if it's a directory
        if os.path.isdir(second_level_path):
            # Iterate through .mat files in the second-level directory
            for file_name in os.listdir(second_level_path):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(second_level_path, file_name)
                    # Load the .mat file
                    mat_data = scio.loadmat(file_path)
                    G = nx.from_numpy_matrix((mat_data['G']))
                    G1 = G.to_undirected()
                    G1.remove_edges_from(nx.selfloop_edges(G1))
                    G_dict[second_level_folder]=G1
    network_name=list(G_dict.keys())
    result_dict = defaultdict(list)
    for id1,value1 in (enumerate(network_name)):
        G1=G_dict[value1]
        for id2,value2 in (enumerate((network_name))):
            G2 = G_dict[value2]
            other_dict=nh.similarity_2g_various_size(G1, G2)
            a1, b1 = nh.NND_GAP_cross3(G1)
            a2, b2 = nh.NND_GAP_cross3(G2)
            other_dict['our']=math.sqrt((a1-a2)**2+(b1-b2)**2)
            for key,value in other_dict.items():
                result_dict[key].append((value1,value2,value))

    return result_dict

def read_networks_distance_14(base_directory,first_level_folder):
    first_level_path = os.path.join(base_directory, first_level_folder)
    G_dict={}
    for second_level_folder in tqdm(os.listdir(first_level_path)):
        second_level_path = os.path.join(first_level_path, second_level_folder)
        G1=nh.real_networks(second_level_path)
        G_dict[second_level_folder]=G1
    network_name=list(G_dict.keys())
    result_dict = defaultdict(list)
    for id1,value1 in (enumerate(network_name)):
        G1=G_dict[value1]
        for id2,value2 in (enumerate((network_name))):
            G2 = G_dict[value2]
            other_dict=nh.similarity_2g_various_size(G1, G2)
            a1, b1 = nh.NND_GAP_cross3(G1)
            a2, b2 = nh.NND_GAP_cross3(G2)
            other_dict['our']=math.sqrt((a1-a2)**2+(b1-b2)**2)
            for key,value in other_dict.items():
                result_dict[key].append((value1,value2,value))
    return result_dict

def read_173networks_distance():
    base_directory='./network-similarity/dataset/'
    distance_data = []
    cpu_cores = 23
    pool = Pool(cpu_cores)
    results = []
    for first_level_folder in tqdm(os.listdir(base_directory)):
        results.append(pool.apply_async(read_networks_distance, args=(base_directory, first_level_folder)))
    pool.close()
    pool.join()
    for result in [r.get() for r in results]:
        distance_data.append(result)
    with open('./network-similarity/real_networks_distance.pkl', 'wb') as file:
        pickle.dump(distance_data, file)


def read_14networks_distance():
    base_directory='./network-similarity/dataset/'
    distance_data = []
    cpu_cores = 5
    pool = Pool(cpu_cores)
    results = []
    for first_level_folder in tqdm(os.listdir(base_directory)):
        results.append(pool.apply_async(read_networks_distance_14, args=(base_directory, first_level_folder)))
    pool.close()
    pool.join()
    print('done')
    for result in [r.get() for r in results]:
        distance_data.append(result)
    with open('./network-similarity/real_networks_distance_14.pkl', 'wb') as file:
        pickle.dump(distance_data, file)
def dk_model_similarity():
    import os
    file_base='./network-similarity/dataset/dk'
    folder1_list = ["dk2.0","dk2.1","dk2.5"]
    folder2 = "./network-similarity/dataset/original"
    prefixes = ['dk2.0_', 'dk2.1_', 'dk2.5_']
    similarity_dict = {}
    similarity_dict_initial={}
    for i1 in folder1_list:
        folder1=file_base+'/'+i1
        files1 = os.listdir(folder1)
        files2 = os.listdir(folder2)
        files1_no_prefix = {}
        for file in files1:
            for prefix in prefixes:
                if file.startswith(prefix):
                    stripped_name = file[len(prefix):]
                    files1_no_prefix[stripped_name] = file
                    break
        for file2 in tqdm(files2):
            if file2 in files1_no_prefix:
                file2_path = os.path.join(folder1, files1_no_prefix[file2])
                file1_path = os.path.join(folder2, file2)
                G_random=nh.read_txt_new(file2_path)
                G_initial = nh.read_txt_new(file1_path)
                a, b = nh.NND_GAP_cross3(G_initial)
                similarity_dict_initial[(file2, i1)] = [a, b, 0]
                a1, b1 = nh.NND_GAP_cross3(G_random)
                similarity_dict[(file2,i1)]=[a1,b1,math.sqrt((a - a1) ** 2 + (b - b1) ** 2)]
    with open('./network-similarity/real_networks_dk.pkl', 'wb') as file:  # 'wb' 表示以二进制写入模式打开文件
        pickle.dump(similarity_dict, file)

def dk_model_similarity_distance():
    import os
    # 定义文件夹路径
    file_base='./network-similarity/dataset/dk'
    folder1_list = ["dk2.0","dk2.1","dk2.5"]
    folder2 = "./network-similarity/dataset/original"
    prefixes = ['dk2.0_', 'dk2.1_', 'dk2.5_']
    similarity_dict = {}
    for i1 in folder1_list:
        folder1=file_base+'/'+i1
        files1 = os.listdir(folder1)
        files2 = os.listdir(folder2)
        files1_no_prefix = {}
        for file in files1:
            for prefix in prefixes:
                if file.startswith(prefix):
                    stripped_name = file[len(prefix):]
                    files1_no_prefix[stripped_name] = file
                    break
        for file2 in tqdm(files2):
            if file2 in files1_no_prefix:
                file2_path = os.path.join(folder1, files1_no_prefix[file2])
                file1_path = os.path.join(folder2, file2)
                G_random=nh.read_txt_new(file2_path)
                G_initial = nh.read_txt_new(file1_path)
                other_dict = nh.similarity_2g_various_size(G_random, G_initial)
                a, b = nh.NND_GAP_cross3(G_initial)
                a1, b1 = nh.NND_GAP_cross3(G_random)
                other_dict['our']=math.sqrt((a - a1) ** 2 + (b - b1) ** 2)
                similarity_dict[(file2,i1)]=other_dict
    with open('./network-similarity/real_networks_dk_distance.pkl', 'wb') as file:
        pickle.dump(similarity_dict, file)

def draw_real_networks():
    files='.\\real_networks_dk_distance.pkl'
    with open(files, 'rb') as file:
        dk_distance = pickle.load(file)
    network_name=[]
    method_name=[]
    value_list=[]
    dk_list=[]
    filter_network=['animal-social.txt','bio.txt','brain.txt','DIMACS.txt','email.txt','INTERACTION.txt','RETWEET.txt','road.txt']

    for key,ivalue in dk_distance.items():
        if key[0] in filter_network:
            continue
        else:
            network_name.extend([key[0]]*len(list(ivalue.keys())))
            dk_list.extend([key[1]]*len(list(ivalue.keys())))
            method_name.extend(ivalue.keys())
            value_list.extend((ivalue.values()))
    df = pd.DataFrame({
        'network_name': network_name,
        'method_name': method_name,
        'dk': dk_list,
        'value_list': value_list
    })
    from matplotlib import rcParams
    rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for imethod in list(set(method_name)):
        df_per=df[df.method_name==imethod]
        pivot_df = df_per.pivot(index='network_name', columns='dk', values='value_list')
        result_array = pivot_df.to_numpy()
        ax = axes[int(list(set(method_name)).index(imethod)/3), int(list(set(method_name)).index(imethod)%3)]
        g = sns.heatmap(pivot_df, ax=ax,cmap='RdYlBu_r', cbar=True)
        g.set_facecolor('xkcd:white')
        ax.set_title(imethod)
    plt.tight_layout()
    plt.show()
    dk_values = df['dk'].unique()
    for dk in dk_values:
        filtered_df = df[df['dk'] == dk]
        pivot_df = filtered_df.pivot(index='network_name', columns='method_name', values='value_list')
        output_file_name = f'.\dk_{dk}.csv'
        pivot_df.to_csv(output_file_name, index=True)



def networks_spreading():
    first_level_folder = '.\data'
    # Walk through the second-level directory
    real_tau_all={}
    for second_level_folder in tqdm(os.listdir(first_level_folder)):
        second_level_path = os.path.join(first_level_folder, second_level_folder)
        if second_level_path.endswith('.txt'):
            G = nh.read_txt_new(second_level_path)
        if second_level_path.endswith('.csv'):
            G = nh.real_networks(second_level_path)
        G1 = G.to_undirected()
        G1.remove_edges_from(nx.selfloop_edges(G1))
        NND_node, _, _, _ = nh.node_edge_measure_Cal(G)
        average_NND=np.average(list(NND_node.values()))
        a1,b1 =nh.NND_GAP_cross3(G1)
        real_tau_all[second_level_folder] = [a1,b1]
    with open('.\\networks_spreading.pkl', 'wb') as file:
        pickle.dump(real_tau_all, file)
















