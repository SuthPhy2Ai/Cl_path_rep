import sys
import os
import json
import numpy as np
import pandas as pd
import random
from ase.io import read
from ase.db import connect
import spglib
from gensim.models import Word2Vec
from mendeleev import element

# Constants
NUM_SLICE = 10
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(parent_dir)
SRC_DIR = os.path.join(parent_dir, "src")
# Load necessary datasets
space_group_df = pd.read_csv(os.path.join(SRC_DIR, "point_group_array.csv"))
w2v_model = Word2Vec.load(os.path.join(SRC_DIR, "mat2vec-master/mat2vec/training/models/pretrained_embeddings"))
atoms_vec = json.load(open(os.path.join(SRC_DIR, "atom_init.json"), "r"))
element_data_df = pd.read_json(os.path.join(SRC_DIR, "allrs.json"), orient='index')

def get_value_by_element(df, element_symbol):
    """Retrieve a value from the dataframe based on the element symbol."""
    if element_symbol in df[0].values:
        return df[df[0] == element_symbol].iloc[0, 1]
    return None

def tsp_dijkstra(data, start_node):
    n = data.shape[0]
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i, j] = np.linalg.norm(data[i] - data[j])
    path = {start_node}
    curr_index = start_node
    distances = []
    predecessors = []
    while len(path) < n:
        unvisited = set(range(n)) - path
        min_index = min(unvisited, key=lambda x: graph[curr_index, x])
        path.add(min_index)
        distances.append(graph[curr_index, min_index])
        predecessors.append(min_index)
        curr_index = min_index
    last_to_start_distance = graph[curr_index, start_node]
    total_distance = sum(distances) + last_to_start_distance
    path_nodes = [start_node] + predecessors
    
    return total_distance, path_nodes

def tsp_greed(data, start_node):
    """Solve the Traveling Salesman Problem (TSP) using a greedy heuristic."""
    n = data.shape[0]
    graph = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i, j] = np.linalg.norm(data[i] - data[j])
    
    path = {start_node}
    curr_index = start_node
    distances, predecessors = [], []
    
    while len(path) < n:
        min_distance, min_index = np.inf, -1
        for i in range(n):
            if i not in path and graph[curr_index, i] < min_distance:
                min_distance = graph[curr_index, i]
                min_index = i
        if min_index != -1:
            path.add(min_index)
            curr_index = min_index
            distances.append(min_distance)
            predecessors.append(curr_index)
    
    last_to_start_distance = graph[curr_index, start_node]
    total_distance = sum(distances) + last_to_start_distance
    path_nodes = [start_node] + list(predecessors)
    
    return total_distance, path_nodes


def build_dist_matrix(data):
    """Construct a distance matrix based on Euclidean distances."""
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(data[i] - data[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

def get_embed_dist(dist_matrix, lst, win):
    """Compute node distances using a sliding window approach."""
    dist_lst = []
    for idx in range(len(lst)):
        if idx < len(lst) - win:
            tmp_lst = [lst[idx], lst[idx + win]]
        else:
            tmp_lst = [lst[idx], lst[idx - win]]
        dist = dist_matrix[tmp_lst[0], tmp_lst[1]]
        dist_lst.append(min(1 / (dist**2), 1))
    return dist_lst

def get_embed_mat2v_global(formula="BaTiO3"):
    """Retrieve the element embedding vector using the mat2vec method."""
    return w2v_model.wv.get_mean_vector(formula)

def gaussian_basis_functions(d):
    """Expand distances using Gaussian basis functions."""
    mu_values = np.linspace(0.2, 1.7, 24)
    gamma = 10
    return np.exp(-gamma * (d - mu_values) ** 2)

def get_crystal_path_muhead(ase_obj=True, 
                            stru=None, 
                            ids=None, 
                            all_files=None, 
                            num=None, 
                            work_path=None, 
                            mpid=None, 
                            elec=None, 
                            nebmax=4, 
                            num_heads=4):
    """Process the structure and compute features for randomly selected atomic heads."""
    if not ase_obj:
        cif_path = all_files[num]
        ids = all_files[num].split('/')[-1].split('.')[0]
        stru = read(cif_path)
    # 
    # if len(stru) > 30 or len(stru) < 4:
    #     raise ValueError(f"Invalid number of atoms: {len(stru)}")
    if len(stru) < 4:
        stru = stru * (2, 2, 2) # duplicate the unit cell to make sure there are enough atoms
    if len(stru) > 30:
        #raise ValueError(f"TOO MANY number of atoms in unit cell: {len(stru)}")
        pass
    # print("there is {} atoms in the unit cell".format(len(stru)))
    selected_heads = random.sample(range(len(stru)), min(num_heads, len(stru)))
    Atom_feature = []
    for head in selected_heads:
        atom_feature = []
        elem_list = stru.get_chemical_symbols()
        node_list = stru.get_atomic_numbers()
        point_set = stru.get_positions()
        formula = stru.get_chemical_formula()
        data = point_set.copy()
        distance, path = tsp_dijkstra(data, head)
        dist_matrix = build_dist_matrix(data)
        rs__ = np.zeros((24, len(path)))
        for m in [elem_list[i] for i in path]:
            rs = get_value_by_element(element_data_df, m)
            rs__[:, list([elem_list[i] for i in path]).index(m)] = gaussian_basis_functions(rs)
        dist__ = np.zeros((nebmax, len(path)))
        for i in range(1, nebmax + 1):
            dist__[i-1, :] = get_embed_dist(dist_matrix, path, win=i)
        mat2v_global = get_embed_mat2v_global(formula)
        spacegroup_info = spglib.get_symmetry_dataset(stru)
        # print("spacegroup_info",spacegroup_info)
        
        pg = spacegroup_info["pointgroup"]
        sym2v_global = np.array(space_group_df[space_group_df["point_group"] == pg])[0][1:].astype(int)
        stru.info.update({"rs__": rs__, 
                          "mat2v_global": mat2v_global, 
                          "dist__": dist__, 
                          "path_indx": node_list[path], 
                          "sym2v_global": sym2v_global})
        for i, k in enumerate(list(node_list[path])):
            tmp_feature = np.concatenate([atoms_vec[str(k)], mat2v_global, sym2v_global, dist__[:, i], rs__[:, i]])
            atom_feature.append(tmp_feature)
        # feat_db = connect(f"{work_path}/deal_json.db")
        # feat_db.write(stru, ids=ids, head=head, atom_feature=json.dumps(atom_feature.tolist()), mpid=mpid, data={'atom_feature': atom_feature, 'elec': np.array(elec)})
        Atom_feature.append(atom_feature)
    return np.array(Atom_feature)


if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset, DataLoader
    from ase.db import connect
    import numpy as np

    class CrystalDataset(Dataset):
        def __init__(self, db_path):
            self.db = connect(db_path)
            self.entries = list(self.db.select())

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            tmp = self.entries[idx]
            atoms = tmp.toatoms()
            try:
                atom_feature = get_crystal_path_muhead(ase_obj=True, stru=atoms)
                target = torch.tensor(tmp.data['dielectric'], dtype=torch.float32)
                atom_feature = torch.tensor(atom_feature, dtype=torch.float32)
                return atom_feature, target
            except ValueError as e:
                print(f"Skipping structure {idx} due to error: {e}")
                return None 


    def collate_fn(batch):
        """
        Custom collate function that handles skipping None samples.
        """
        batch = [b for b in batch if b is not None] 
        if len(batch) == 0:
            return None 
        atom_features, targets = zip(*batch)
        max_atoms = max(feat.shape[1] for feat in atom_features)
        batch_size = len(atom_features)
        num_heads, embed_dim = atom_features[0].shape[0], atom_features[0].shape[2]

        padded_features = torch.zeros((batch_size, num_heads, max_atoms, embed_dim), dtype=torch.float32)
        attention_masks = torch.zeros((batch_size, num_heads, max_atoms), dtype=torch.float32)
        for i, feat in enumerate(atom_features):
            num_atoms = feat.shape[1]
            padded_features[i, :, :num_atoms, :] = feat
            attention_masks[i, :, :num_atoms] = 1 
        targets = torch.stack(targets)
        return padded_features, attention_masks, targets

    from torch.utils.data import DataLoader
    dataset = CrystalDataset("/data/home/hzw1010/suth/elec_gw/dbs/dielectric_with_gap.db")
    dataloader = DataLoader(dataset, batch_size=64,  num_workers=2,pin_memory=False,
                            shuffle=True, collate_fn=collate_fn, drop_last=False)
    from tqdm.auto import tqdm
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    import time
    start_time = time.time()
    for batch_features, batch_masks, batch_targets in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        print("batch_features,batch_masks,batch_targets",batch_features.shape, batch_masks.shape, batch_targets.shape)
        break
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")