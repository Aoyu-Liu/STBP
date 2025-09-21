import torch
from torch_geometric.data import Data, Dataset

class SpatioTemporalDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] 
            self.y = inputs[split+'_y'] 
        else:
            self.x = x
            self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        return Data(x=x, y=y) 
    
class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return Data(x=x)