import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        return input


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1,1))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1,1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Conv(nn.Module):
    def __init__(self, features, dropout=0.2):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class DLGA(nn.Module): 
    def __init__(self, d_model, head, seq_length=1):
        super(DLGA, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

    def forward(self, input, node_emb):
        _, d_model, num_nodes, seq_length = input.shape
        query, key, value = self.q(input), self.k(input), self.v(input)
        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        key = key.view(
            key.shape[0], -1, self.d_k, key.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)  
        node_emb = node_emb.view(
            node_emb.shape[0], -1, self.d_k, node_emb.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)

        key = torch.softmax(key / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        node_emb = torch.softmax(node_emb / math.sqrt(self.d_k), dim=-1)

        kv = torch.einsum("bhlnx, bhlny->bhlxy", key, value)
        attn_qkv1 = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)

        node_emb_v = torch.einsum("bhlnx, bhlny->bhlxy", node_emb, value)
        attn_qkv2 = torch.einsum("bhlnx, bhlxy->bhlny", query, node_emb_v)

        attn_qkv = attn_qkv1 + attn_qkv2

        x = (
            attn_qkv.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(attn_qkv.shape[0],d_model, num_nodes, seq_length)
        )
        x = self.concat(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Encoder(nn.Module):
    def __init__(self, d_model, head, seq_length=1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model

        self.attn = DLGA(
            d_model, head, seq_length
        )

        self.FFW = Mlp(d_model)

        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
    
    def forward(self, x, node_emb):
        gate, scale, memory = torch.chunk(node_emb, 3, dim=1)
        x = x + gate * self.attn(modulate(self.norm1(x), 0, scale), memory)
        x = x + gate * self.FFW(modulate(self.norm2(x), 0, scale))
        return x


class FConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, fc=True):
        super(FConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        self.fre_network = FreNetwork(out_features)
        if not fc:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x):
        input_x = x              
        output = self.fre_network(self.weight_neigh(input_x))       
        if self.weight_self is not None:
            output += self.weight_self(x)  
        return output


class FreNetwork(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.complex_weight = nn.Parameter(torch.randn(1, 1, time_dim//2 + 1, 2, dtype=torch.float32) * 0.02)
        nn.init.xavier_uniform_(self.complex_weight)

    def forward(self, x):
        B, N, T = x.shape
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        t_bank= torch.view_as_complex(self.complex_weight)
        x = x * t_bank
        x = torch.fft.irfft(x, n=self.time_dim, dim=2, norm='ortho')
        return x
    
    
class STBP_Model(nn.Module):
    """Some Information about EAC_Model"""
    def __init__(self, args):
        super(STBP_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        self.fconv1 = FConv(args.model["in_channel"], args.model["hidden_channel"], bias=True, fc=False)
        self.stmodule = Encoder(
            d_model=args.model["hidden_channel"],
            head=1,
            seq_length=1,
        )
        self.fconv2 = FConv(args.model["hidden_channel"], args.model["out_channel"], bias=True, fc=False)
        self.fc = nn.Linear(args.model["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        self.pattern_bank = nn.Parameter(torch.empty(args.base_node_size, args.model["hidden_channel"]*3).uniform_(-0.1, 0.1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.args.model["hidden_channel"]*3, dtype=self.pattern_bank.dtype, device=self.pattern_bank.device).uniform_(-0.1, 0.1))
            self.pattern_bank = nn.Parameter(torch.cat([self.pattern_bank, new_params], dim=0))
            self.num_nodes = new_num_nodes

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.model["in_channel"]))  
        B, N, T = x.shape
        x = F.relu(self.fconv1(x))                              

        adaptive_params = self.pattern_bank 
        
        node_emb = adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape).permute(0,2,1).unsqueeze(-1)  # [bs, N, feature]
        x = x.permute(0,2,1).unsqueeze(-1)

        x = self.stmodule(x, node_emb).squeeze(-1)
        x = x.permute(0,2,1).squeeze(-1)
        x = self.fconv2(x)
        x = x.reshape((-1, self.args.model["out_channel"]))         
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
