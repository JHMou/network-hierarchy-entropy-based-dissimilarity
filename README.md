
# Network Hierarchy Entropy-based Network Dissimilarity

This repository implements "Network Hierarchy Entropy for Quantifying Graph Dissimilarity" using Python, R, and C++.

## Datasets

1. **Synthetic networks**: Generated using network models with parameters from the paper
2. **General empirical networks**: `Networks.zip`
3. **Temporal mobility networks**: `Temporal mobility networks.zip`
4. **Biological macromolecule networks**: `Enzymes.zip`
5. **dk randomized networks**: `empirical-DK-50.zip`
6. **Example networks**: karate, synapse, polbooks, and london networks for node hierarchy structure preservation

## Package Installation

```bash
pip install dotmotif
pip install netrd
```

## Figure Reproductions

### Figure 1

#### Fig. 1(b)

**Experiment Results:**

```bash
python SIR_model_node.py
python SIR_model_edge.py
```

- Spreading influence results saved in:
  - `SIR results` (nodes)
  - `SIR results-edge` (edges)
- Kendall tau coefficients saved as:
  - `NHC_tau.csv` (node hierarchy centrality)
  - `EHC_tau.csv` (edge hierarchy centrality)

**Visualization:**

```bash
python NHC2R.py
Rscript R4fig1b.R
```

#### Fig. 1(d)

**Tij and Tk Generation:**

```bash
python NHE4nodes.py
```

*Visualization created with Gephi*

**Relationship Analysis:**

```bash
python NHE2infection.py
```

### Figure 2

**Network Models and Similarity Matrices:**

```bash
python synthetic_network_cluster.py
```

### Figure 3

#### Fig. 3(a)

**DK Network Distance Analysis:**

```bash
python dk_average_distance.py
```

*Output: `R_dk_boxplot_data.csv` for Fig. S9*

#### Fig. 3(b)

**Network Visualization:**

```bash
cd LaNet-vi_2.1.3
./lanet -input dk2.0_chemical.txt -render svg -opacity 0.3 -W 2400 -H 1800
```

### Figure 4

#### Fig. 4(a)

**Network Generation:**

```bash
cd Simulated Annealing
./RandNetGen -net karate.txt -knn original
```

*Networks saved to `./Fig.4/karate`*

#### Fig. 4(b-d)

**Analysis and Visualization:**

```bash
python SA4karate.py
```

Key functions in `SA4karate.py`:
- `draw_historty_3d()`: 3D network evolution visualization
- `draw_karate()`: Network comparison visualizations
- `filiter_networks_k()`: Network filtering by k-value
- `draw_k_class()`: Similarity matrix visualization

### Figure 5

**Mobility Network Analysis:**

```bash
python network_similarity-mobility.py
```

Key functions:
- `SLouvain_networks()`: Community detection (Fig. 5a)
- `china_mobility_parallel()`: Experiment results (Fig. 5b, 5d)
- `draw_scatter_line_nonconsective()`: Visualization (Fig. 5b)
- `network_centralities_china()`: Visualization (Fig. 5c)
- `draw_scatter_line_nonconsective_others()`: Visualization (Fig. 5d)

### Figure 6

**Network Classification:**

```bash
python network_similarity-classification.py
```

Key functions:
- `draw_graph_classification()`: Group difference analysis (Fig. 6a)
- `draw_motif_DD()`: Motif distribution (Fig. 6b)
- `graph_classification_heatmap()`: Classification accuracy (Fig. 6c)

**Heatmap Visualization:**

```bash
Rscript heatmap.R
```

## Supplementary Figures

### Fig. S1

```bash
python NHE_EHE-simple_networks.py
```

### Fig. S2

```bash
python NHE_time.py
```

### Fig. S3

```bash
python network_similarity-mobility-consecutive.py
```

*Use functions: `china_mobility_parallel()` with `draw_scatter_line_combined()`*

### Fig. S4

```bash
python network_similarity-mobility-full.py
```

*Use functions: `china_mobility_parallel()` with `draw_distance_mobility_combined()`*

### Fig. S5

```bash
python NHC-similarity.py
```

### Fig. S6

```bash
python NHC-dissimilarity-chemical.py
```

### Fig. S7

```bash
python network_similarity-mK5.py
```

### Fig. S8

```bash
python network_similarity-synthetic.py
```

Key functions:
- `WS_BA_ER_distance_station()`: Network generation and distance calculation
- `draw_model_ws()`: Distance heatmap visualization

### Fig. S9

```bash
Rscript BOXPLOT-significance-dk.R
```

### Fig. S10

```bash
python dk_clustering.py
```

### Fig. S11-S13

**Hierarchy-preserving Networks:**

```bash
python greedy_edge_swap.py
```

**Network Dissimilarity:**

```bash
python network-dissimilarity-hierarchy-preserving.py
```

Key functions:
- `filiter_networks_k()`: Network filtering by distance distribution
- `draw_filiter_networks_k_chemical()`: Figure reproduction
```