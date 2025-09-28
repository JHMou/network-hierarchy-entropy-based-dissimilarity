import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

colorbar = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', "#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"]
class LFR_network():
    def __init__(self,  *args, **kwargs):
        if args:
            n,tau1,tau2,mu,average_d,max_d,min_community,max_community,seed = args
            self.n = n
            self.tau1 = tau1
            self.tau2 = tau2
            self.seed = seed
            self.mu = mu
            self.average_d = average_d
            self.max_d = max_d
            self.min_community = min_community
            self.max_community = max_community
        else:
            self.n = kwargs["n"]
            self.tau1 = kwargs["tau1"]
            self.tau2 = kwargs["tau2"]
            self.seed = kwargs["seed"]
            self.mu = kwargs["mu"]
            self.average_d = kwargs["average_d"]
            self.max_d = kwargs["max_d"]
            self.min_community = kwargs["min_community"]
            self.max_community = kwargs["max_community"]

    def init_network(self,max_attempts=100):
        from networkx.generators.community import LFR_benchmark_graph
        attempts = 0
        while attempts < max_attempts:
            self.G = LFR_benchmark_graph(n=self.n, tau1=self.tau1, tau2=self.tau2, mu=self.mu, average_degree=self.average_d, max_degree=self.max_d,
                                         min_community=self.min_community, max_community=self.max_community, seed=self.seed)
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
            if nx.is_connected(self.G):
                return self.G
            else:
                self.seed += 1
                attempts += 1
        raise RuntimeError(f"Unable to generate a connected graph after {max_attempts} attempts.")

    def get_LFRcom_label(self):
        self.G.communities = {frozenset(self.G.nodes[v]["community"]) for v in self.G}
        com_num = len(self.G.communities)
        self.G.node_com_dic = {}
        num = 0
        for group in self.G.communities:
            self.G.node_com_dic[num] = list(group)
            num += 1
        return self.G.node_com_dic

def LFR_init(*args):
    LFR = LFR_network(*args)
    LFR_graph = LFR.init_network()
    LFR_label = LFR.get_LFRcom_label()
    LFR_true_label = []
    for i in LFR_graph.nodes:
        for j in LFR_label.keys():
            if i in LFR_label[j]:
                LFR_true_label.append(j + 1)
                break
    return LFR_graph, LFR_true_label

def network_visual(G):
    if True:
        com_num = len(G.communities)
        use_colors = random.sample(colorbar, com_num)
        colors = []
        for i in G.nodes:
            for j in G.node_com_dic.keys():
                if i in G.node_com_dic[j]:
                    colors.append(use_colors[j])
                    break

        degrees = dict(G.degree)
        figlength = max(6,13 * int(len(G.nodes) / 150))
        plt.figure(figsize=(figlength,figlength))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=colors,
                with_labels=True)
        plt.show()

if __name__ == "__main__":


    mu_list = np.arange(1,15)
    for mu in mu_list:
        paras = {"n":100, "tau1":3, "tau2":2, "mu":mu/20, "average_d":4, "max_d":20, "min_community":20, "max_community":50, "seed":42}
        args = tuple(paras.values())
        G, true_label = LFR_init(*args)
        network_visual(G)

