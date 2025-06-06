import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
from chemprop import utils
import networkx as nx


# Get additional atom features for each molecule to specify the terminal atoms
# This is an n×1 numpy array where n is the number of atoms. Terminal atoms and their adjacent neighbors are marked as 1, second nearest neighbors as 0.5, and others as 0.
def calc_terminal_features(smiles):
    # chemprop.utils.make_mol ~ rdkit.Chem.MolFromSmiles
    rdmol = utils.make_mol(smiles, keep_h=False, add_h=False)
    # When calculate the distance between nodes, atoms in one ring were treated as one node.
    ssr = [set(x) for x in Chem.GetSymmSSSR(rdmol)]
    i = 0
    while i < len(ssr):
        ring_member = ssr[i]
        j = i + 1
        while j < len(ssr):
            ring_member_next = ssr[j]
            if ring_member.intersection(ring_member_next):
                ring_member.update(ring_member_next)
                ssr.pop(j)
            else:
                j += 1
        i += 1     
    # Get node id mapping between original graph and new graph without ring
    ori_new_node_id_map = {}
    new_ori_node_id_map = {}
    for ring_id, ring_members in enumerate(ssr):
        for at_id in ring_members:
            ori_new_node_id_map[at_id] = ring_id
        new_ori_node_id_map[ring_id] = ring_members
    # Construct new graph
    condensed_graph = nx.Graph()
    n_atoms = rdmol.GetNumAtoms()
    node_id_count = len(ssr)
    for at_id in range(n_atoms):
        if not rdmol.GetAtomWithIdx(at_id).IsInRing():
            ori_new_node_id_map[at_id]=node_id_count
            new_ori_node_id_map[node_id_count] = {at_id}
            node_id_count += 1
    condensed_graph.add_nodes_from(new_ori_node_id_map.keys())
    for bond in rdmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        node1 = ori_new_node_id_map[a1]
        node2 = ori_new_node_id_map[a2]
        if node1 != node2:
            condensed_graph.add_edge(node1, node2)
    # Get a pair of nodes with longest distance in graph (which are specified as terminals)
    paths = nx.shortest_path_length(condensed_graph)
    longest_path = {}
    for node0, path_len_dict in nx.shortest_path_length(condensed_graph):
        node1 = max(path_len_dict, key=path_len_dict.get)
        longest_path[(node0, node1)] = path_len_dict[node1]
    terminal_nodes = max(longest_path, key=longest_path.get)

    def get_atom_features(terminal_node):
        feature = np.zeros((n_atoms, 1))
        node_distance = nx.single_source_shortest_path_length(condensed_graph, terminal_node)
        atom_ids_1 = set()
        atom_ids_2 = set()
        for node, dist in node_distance.items():
            if dist <= 1:
                atom_ids_1.update(new_ori_node_id_map[node])
            elif dist == 2:
                atom_ids_2.update(new_ori_node_id_map[node])
        feature[[*atom_ids_1],]=1
        feature[[*atom_ids_2],]=0.5
        return feature, atom_ids_1 | atom_ids_2
    
    term0_feature, term0_atom_ids = get_atom_features(terminal_nodes[0])
    term1_feature, term1_atom_ids = get_atom_features(terminal_nodes[1])

    return (term0_feature, term0_atom_ids), (term1_feature, term1_atom_ids)


