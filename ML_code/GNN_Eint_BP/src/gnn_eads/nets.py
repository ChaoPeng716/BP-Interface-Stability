"""Module containing the Graph Neural Network classes."""
import os.path as osp

import torch
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GraphMultisetTransformer
from torch_geometric.data import Data

from gnn_eads.constants import NODE_FEATURES
from gnn_eads.functions import get_graph_conversion_params, get_mean_std_from_model
    
    
class FlexibleNet(torch.nn.Module):
    def __init__(self, 
                 dim: int=128,                  
                 N_linear: int=1,
                 N_conv: int=3,
                 adj_conv: bool=False,
                 in_features: int=NODE_FEATURES,                 
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv=SAGEConv,
                 pool=GraphMultisetTransformer, 
                 pool_ratio: float=0.25, 
                 pool_heads: int=4, 
                 pool_seq: list[str]=["GMPool_3G", "SelfAtt", "GMPool_I"], 
                 pool_layer_norm: bool=False):
        """Flexible Net for defining multiple GNN model architectures.

        Args:
            dim (int, optional): Layer width. Defaults to 128.
            N_linear (int, optional): Number of dense. Default to 3.
            N_conv (int, optional): Number of convolutional layers. Default to 3.
            adj_conv (bool, optional): Whether include linear layer between each convolution. Default to True.
            in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
        """
        super(FlexibleNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.sigma = sigma
        self.conv = conv
        self.num_conv_layers = N_conv
        self.num_linear_layers = N_linear
        self.adj_conv = adj_conv        
        # Building blocks of the GNN
        self.input_layer = Linear(self.in_features, self.dim, bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_linear_layers)])
        self.conv_block = torch.nn.ModuleList([conv(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        self.pool = pool(self.dim, self.dim, 1, num_nodes=300,         
                         pooling_ratio=pool_ratio, pool_sequences=pool_seq,
                         num_heads=pool_heads, layer_norm=pool_layer_norm)                                                             
        
    def forward(self, data):
        #------------#
        # NODE LEVEL #
        #------------#        
        out = self.sigma(self.input_layer(data.x))  # Input layer
        # Dense layers 
        for layer in range(self.num_linear_layers):
            out = self.sigma(self.linear_block[layer](out))
        # Convolutional layers
        for layer in range(self.num_conv_layers):
            if self.adj_conv:
                out = self.sigma(self.adj_block[layer](out))
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        out = self.pool(out, data.batch, data.edge_index)
        return out.view(-1)

class PreTrainedModel():
    def __init__(self, model_path: str):
        """Container class for loading pre-trained GNN models on the cpu.
        Args:
            model_path (str): path to model folder. It must contain:
                - model.pth: the model architecture
                - GNN.pth: the model weights
                - performance.txt: the model performance and settings                
        """
        self.model_path = model_path
        self.model = torch.load("{}/model.pth".format(self.model_path),
                                map_location=torch.device("cpu"))
        self.model.load_state_dict(torch.load("{}/GNN.pth".format(self.model_path), 
                                              map_location=torch.device("cpu")))
        self.model.eval()  # Inference mode
        self.model.to("cpu")
        # Scaling parameters (only standardization for now)
        self.mean, self.std = get_mean_std_from_model(self.model_path)
        # Graph conversion parameters
        self.g_tol, self.g_sf, self.g_metal_2nn = get_graph_conversion_params(self.model_path)
        # Model info
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
               
        param_size, buffer_size = 0, 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        self.size_all_mb = (param_size + buffer_size) / 1024**2
        
    def __repr__(self) -> str:
        string = "Pretrained graph neural network for DFT ground state energy prediction."
        string += "\nModel path: {}".format(osp.abspath(self.model_path))
        string += "\nNumber of parameters: {}".format(self.num_parameters)
        string += "\nModel size: {:.2f} MB".format(self.size_all_mb)
        return string
    
    def evaluate(self, graph: Data) -> float:
        """Evaluate graph energy

        Args:
            graph (Data): adsorption/molecular graph

        Returns:
            float: system energy in eV
        """
        # return self.model(graph).y.item()
        return self.model(graph).item() * self.std + self.mean    