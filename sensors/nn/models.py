import numpy as np

import math
import torch

from sklearn.cluster import KMeans
from collections import OrderedDict

from torch import nn
from torch.nn import Linear, Conv1d, LayerNorm, DataParallel, ReLU, Sequential
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn.dense import mincut_pool, dense_mincut_pool
from torch_geometric.utils import dense_to_sparse
from torch.nn.functional import glu

class ClusterTS(nn.Module):
    def __init__(self,
                 conv1d_n_feats, conv1d_kernel_size, conv1d_stride,
                 graphconv_n_feats,
                 n_timestamps,
                 n_clusters,
                 n_extra_feats,
                 weight_coords=0.5):
        
        super(ClusterTS, self).__init__()

        self.weight_coords = weight_coords

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=conv1d_n_feats,
                                kernel_size=conv1d_kernel_size, stride=conv1d_stride)
        
        self.L_in = n_timestamps
        self.L_out = math.floor((self.L_in - conv1d_kernel_size)/conv1d_stride + 1)

        self.conv1d_out = conv1d_n_feats*self.L_out
        
        mlp_in = self.conv1d_out + n_extra_feats
        self.mcp_mlp = Linear(mlp_in, n_clusters)
    
    def forward(self, X, A, extra_feats=None):

        # Data
        X = X.float()
        norm_X = LayerNorm(X.shape, elementwise_affine=False) # Normalizes the <entire matrix> to 0 mean 1 var
        X = norm_X(X)

        X = X.unsqueeze(1) # adjusting shape for conv1d
        X = self.conv1d(X)

        X = X.reshape((X.shape[0],-1)) #

        if extra_feats is not None:
            norm_f = LayerNorm(extra_feats.shape, elementwise_affine=False)
            extra_feats = self.weight_coords*norm_f(extra_feats)
            X = torch.cat((X,extra_feats),dim=1)

        S = self.mcp_mlp(X)

        _, _, loss_mc, loss_o = dense_mincut_pool(X, A, S)

        # return torch.softmax(S, dim=-1), loss_mc, loss_o
        return S, loss_mc, loss_o
    
    def reset_parameters(self):
        # Reset parameters of Conv1d layer
        self.conv1d.reset_parameters()
        # Reset parameters of Linear layer
        self.mcp_mlp.reset_parameters()

##############################################################################################################    

class Autoencoder(nn.Module):

    # > Conv1d > MLP encoder > MLP decoder > ConvTranspose1d

    def __init__(self,
                 conv1d_n_feats, conv1d_kernel_size, conv1d_stride,
                 n_timestamps,
                 n_encoding_layers,
                 reduction
                 ):
        
        super(Autoencoder, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=conv1d_n_feats,
                                kernel_size=conv1d_kernel_size, stride=conv1d_stride)
        
        self.L_in = n_timestamps
        self.L_out = math.floor((self.L_in - conv1d_kernel_size)/conv1d_stride + 1)


        self.L_out_T = (self.L_out-1)*conv1d_stride + conv1d_kernel_size
        self.convT1d = nn.ConvTranspose1d(in_channels=conv1d_n_feats, out_channels=1,
                                           kernel_size=conv1d_kernel_size, stride=conv1d_stride, 
                                           output_padding=self.L_in-self.L_out_T)       

        mlp_in = conv1d_n_feats*self.L_out

        mlp_in_list = []
        mlp_in_list.append(mlp_in)

        # Encoder
        encoder_layers = OrderedDict()
        encoder_layers['linear_0'] = Linear(mlp_in, round( mlp_in * reduction ))
        encoder_layers['relu_0'] = ReLU()

        for n in range(1, n_encoding_layers):
            mlp_in = round( mlp_in * reduction )
            mlp_in_list.append(mlp_in)

            encoder_layers[f'linear_{n}'] = Linear(mlp_in, round( mlp_in * reduction ))
            encoder_layers[f'relu_{n}'] = ReLU()
        
        mlp_in_list.append( round(mlp_in * reduction ))
        self.encoder = Sequential(encoder_layers)

        # Decoder

        decoder_layers = OrderedDict()
        for n in range(0, n_encoding_layers):
            decoder_layers[f'linear_{n}'] = Linear( mlp_in_list[-n-1], mlp_in_list[-n-2] )
            decoder_layers[f'relu_{n}'] = ReLU()
        
        self.decoder = Sequential(decoder_layers)

    def forward(self, x):

        x = x.unsqueeze(1)
        f = self.conv1d(x)
        f = f.reshape((f.shape[0],-1))

        z = self.encoder(f)
        o = self.decoder(z)

        o = o.reshape((o.shape[0], -1, self.L_out ))
        r = self.convT1d(o)

        r = r.squeeze()

        return r


    def reset_parameters(self):
        self.conv1d.reset_parameters()
        self.convT1d.reset_parameters()
        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()




##############################################################################################################    

def kmeans_features(data, num_clusters):

    def cluster_kmeans(tensor, k):
        kmeans = KMeans(n_clusters=k, n_init=1)
        kmeans.fit(tensor)
        return kmeans.labels_

    kmeans_features = []
    # Perform clustering for each number of clusters
    for k in num_clusters:
        # Perform K-means clustering
        cluster_labels = cluster_kmeans(data, k)
        kmeans_features.append(cluster_labels)

    return torch.tensor(np.array(kmeans_features).T)



####################################
def main():
    return 0


if __name__ == "__main__":
    main()
