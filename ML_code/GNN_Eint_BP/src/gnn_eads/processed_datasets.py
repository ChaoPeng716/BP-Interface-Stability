from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np

from gnn_eads.constants import ENCODER, FAMILY_DICT, METALS
from gnn_eads.graph_filters import global_filter, isomorphism_test
from gnn_eads.functions import get_graph_formula

class HetGraphDataset(InMemoryDataset):
    """
    InMemoryDataset is the abstract class for creating custom datasets.
    Each dataset gets passed a dataset folder which indicates where the dataset should
    be stored. The dataset folder is split up into 2 folders, a raw_dir where the dataset gets downloaded to,
    and the processed_dir, where the processed dataset is being saved.
    In order to create a InMemoryDataset class, four fundamental methods must be provided:
    - raw_file_names(): a list of files in the raw_dir which needs to be found in order to skip the download
    - file_names(): a list of files in the processed_dir which needs to be found in order to skip the processing
    - download(): download raw data into raw_dir
    - process(): process raw_data and saves it into the processed_dir
    """
    def __init__(self,
                 root,
                 identifier: str):
        self.root = str(root)
        self.pre_data = str(root) + "/pre_" + identifier
        self.post_data = str(root) + "/post_" + identifier
        super().__init__(str(root))
        self.data, self.slices = torch.load(self.processed_paths[0])   

    @property
    def raw_file_names(self): 
        return self.pre_data  
    
    @property
    def processed_file_names(self): 
        return self.post_data 
    
    def download(self):
        pass
    
    def process(self):  
        """
        If self.processed_file_names() does not exist, this method is run automatically to process the raw data starting from the path provided in self.raw_file_names() 
        """
        data_list = []   
        dataset_name = self.root.split("/")[-1] 
        with open(self.raw_file_names, 'r') as infile:
            lines = infile.readlines() 
        split_n = lambda x, n: [x[i:i+n] for i in range(0, len(x), n)] 
        splitted = split_n(lines, 5)  # Each sample =  5 text lines   
        for block in splitted:        
            to_int = lambda x: [float(i) for i in x] 
            _, elem, source, target, energy = block  
            atoms_list = elem.split() 
            atoms_count = len(atoms_list) 
            # if dataset_name[:3] != "gas":  # filter for graphs with no metal 
            #     counter = 0
            #     for element in element_list:
            #         if element in METALS:
            #             counter += 1
            #     if counter == 0:
            #         continue                     
            elem_array = np.array(elem.split()).reshape(-1, 1)  
            elem_enc = ENCODER.transform(elem_array).toarray()   
            x = torch.tensor(elem_enc, dtype=torch.float)         
            edge_index = torch.tensor([to_int(source.split()),    
                                       to_int(target.split())],
                                       dtype=torch.long)       
            y = torch.tensor([float(energy)], dtype=torch.float)  # Graph label 
            family = FAMILY_DICT[dataset_name]                    # Chemical family of the adsorbate/molecule 
            data = Data(x=x, edge_index=edge_index, y=y, ener=y, family=family)
            graph_formula = get_graph_formula(data, ENCODER.categories_[0])   
            data = Data(x=x, edge_index=edge_index, y=y, ener=y, family=family, label=_[:-1], atoms_num=atoms_count, formula=graph_formula )  
            
            if global_filter(data):  # To ensure correct adsorbate representation in the graph
                if isomorphism_test(data, data_list):  # To ensure absence of duplicates graphs
                    data_list.append(data)              
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_post_processed_datasets(identifier: str,
                                   paths: dict):
    """Create the graph FG-dataset. 

    Args:
        identifier (str): Graph settings identifier
        paths (dict): Data paths dictionary, each key is the family name (str)
                      and the value a dictionary of paths

    Returns:
        FG_dataset (tuple[HetGraphDataset]): FG_dataset
    """
    
    C_N_dataset = HetGraphDataset(str(paths['C_N']['root']), identifier)
    N_dataset = HetGraphDataset(str(paths['N']['root']), identifier)
    N_N_dataset = HetGraphDataset(str(paths['N_N']['root']), identifier)
    O_dataset = HetGraphDataset(str(paths['O']['root']), identifier)
    P_dataset = HetGraphDataset(str(paths['P']['root']), identifier)
    S_dataset = HetGraphDataset(str(paths['S']['root']), identifier)
    Si_dataset = HetGraphDataset(str(paths['Si']['root']), identifier)
    FG_dataset = (C_N_dataset,
               N_dataset,
               N_N_dataset,
               O_dataset,
               P_dataset, 
               S_dataset,
                Si_dataset) 
    return FG_dataset
