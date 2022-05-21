from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.special import comb
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, Set2Set

from modeling.utils import convert_to_dense, pad_t


class Encoder(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.conv1: GCNConv = GCNConv(node_dim, 16)
        self.conv2: GCNConv = GCNConv(16, 32)
        self.lin1: Linear = Linear(32, 32)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_attr: Tensor,
                batch_mask: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch_mask)
        embeddings = F.relu(self.lin1(x))
        return embeddings


class Decoder(torch.nn.Module):
    def __init__(self,
                 max_nodes_in_graph: int,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 latent_dim: int):
        super(Decoder, self).__init__()
        self.max_nodes_in_graph = max_nodes_in_graph
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.max_edges_in_graph = comb(max_nodes_in_graph + 1, 2, exact=True) - max_nodes_in_graph

        self.lin1: Linear = Linear(32, 16)
        self.lin2: Linear = Linear(16, max_nodes_in_graph * node_dim)

    def forward(self, embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        embeddings = F.relu(self.lin1(embeddings))
        embeddings = F.relu(self.lin2(embeddings)).reshape(
            -1, self.max_nodes_in_graph, self.node_dim)
        return embeddings


class LitAutoEncoder(pl.LightningModule):
    def __init__(self,
                 max_nodes_in_graph: int,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 batch_size: int = 32):
        super().__init__()
        self.max_nodes_in_graph = max_nodes_in_graph
        self.max_edges_in_graph = comb(max_nodes_in_graph + 1, 2, exact=True) - max_nodes_in_graph
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.batch_size = batch_size
        self.encoder: Encoder = Encoder(node_dim, edge_dim, hidden_dim, latent_dim)
        self.decoder: Decoder = Decoder(
            max_nodes_in_graph, node_dim, edge_dim, latent_dim, hidden_dim)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Tensor,
                batch_mask: Tensor) -> Tuple[Tensor, Tensor]:
        embeddings = self.encoder(x, edge_index, edge_weight, batch_mask)
        return self.decoder(embeddings)

    def run_ae(self, batch: Batch):
        embeddings = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        nodes_recon = self.decoder(embeddings)
        return nodes_recon

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        pred_nodes = self.run_ae(batch)
        pred_nodes = pad_t(pred_nodes, dim=2).swapaxes(1, 2)

        target_nodes, target_edges = convert_to_dense(
            batch, self.max_nodes_in_graph, self.batch_size)

        target_nodes = torch.argmax(target_nodes, dim=-1).reshape(self.batch_size, -1)
        bce_loss_nodes = F.cross_entropy(pred_nodes, target_nodes)
        if batch_idx == 1470:
            print(pred_nodes.swapaxes(1, 2)[0])
            print(target_nodes[0])
        bce_loss = bce_loss_nodes
        loss = bce_loss

        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())
