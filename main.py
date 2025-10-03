import torch 
from torch_geometric.data import Data
from mynetwork import DynamicEdgeConv
from gconv import GCNConv




def run_GCNConv(data):
    conv = GCNConv(16, 32)
    out = conv(data.x, data.edge_index)
    return out

def run_DynamicEdgeConv(data):
    conv = DynamicEdgeConv(3, 128, k=6)
    out = conv(data.x, data.edge_index)
    return out

if __name__ == "__main__":
    
    edge_index = torch.tensor([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2,1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())

    out1 = run_GCNConv(data)
    out2 = run_DynamicEdgeConv(data)
