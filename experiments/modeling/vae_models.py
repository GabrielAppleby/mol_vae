from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.special import comb
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from modeling.utils import convert_to_dense, pad_t


class Encoder(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.nn: Sequential = Sequential(Linear(edge_dim, 16), ReLU(), Linear(16, 1))
        self.conv1: GCNConv = GCNConv(node_dim, hidden_dim)
        self.conv2: GCNConv = GCNConv(hidden_dim, hidden_dim)
        self.lin1: Linear = Linear(hidden_dim, hidden_dim)
        self.lin_mu: Linear = Linear(hidden_dim, latent_dim)
        # self.lin_std: Linear = Linear(hidden_dim, latent_dim)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_attr: Tensor,
                batch_mask: Tensor) -> Tuple[Tensor, Tensor]:
        edge_weight = self.nn(edge_attr).squeeze()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = global_mean_pool(x, batch_mask)
        embeddings = F.relu(self.lin1(x))
        mu = self.lin_mu(embeddings)
        # std = self.lin_std(embeddings)
        return mu
        # return mu, std


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

        self.lin1: Linear = Linear(latent_dim, hidden_dim)
        self.lin2: Linear = Linear(hidden_dim, hidden_dim)
        self.lin_out_edges: Linear = Linear(hidden_dim, self.max_edges_in_graph * (edge_dim + 1) + (max_nodes_in_graph * (node_dim + 1)))

    def forward(self, z_sample: Tensor) -> Tuple[Tensor, Tensor]:
        z_sample = F.relu(self.lin1(z_sample))
        z_sample = F.relu(self.lin2(z_sample))
        pred_edges = self.lin_out_edges(z_sample)
        return pred_edges


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

    def _sample(self, z_mu, z_log_var):
        z_std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(z_mu)
        return z_mu + z_std * epsilon

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Tensor,
                batch_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # z_mu, z_log_var = self.encoder(x, edge_index, edge_weight, batch_mask)
        z_mu = self.encoder(x, edge_index, edge_weight, batch_mask)
        # z_sample = self._sample(z_mu, z_log_var)
        # return self.decoder(z_sample)
        return self.decoder(z_mu)

    def run_vae(self, batch: Batch):
        # z_mu, z_log_var = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        z_mu = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # z_sample = self._sample(z_mu, z_log_var)
        edges_recon = self.decoder(z_mu)
        # edges_recon = self.decoder(z_sample)
        # return edges_recon, z_mu, z_log_var
        return edges_recon

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        pred_edges = self.run_vae(batch)
        # pred_edges, z_mu, z_log_var = self.run_vae(batch)
        pred_nodes, pred_edges = torch.split(pred_edges, list([self.max_nodes_in_graph * (self.node_dim + 1), self.max_edges_in_graph * (self.edge_dim + 1)]), dim=1)

        # pred_nodes = pad_t(pred_nodes, dim=2).reshape(-1, self.node_dim + 1)
        pred_nodes = pred_nodes.reshape(self.batch_size, self.node_dim+1, self.max_nodes_in_graph)
        # pred_nodes = pred_nodes.reshape(-1, self.node_dim + 1)
        pred_edges = pred_edges.reshape(self.batch_size, self.edge_dim + 1, self.max_edges_in_graph)
        # pred_edges = pred_edges.reshape(self.batch_size, -1)
        # pred_edges = pad_t(pred_edges, dim=2).reshape(-1, self.edge_dim + 1)

        target_nodes, target_edges = convert_to_dense(
            batch, self.max_nodes_in_graph, self.batch_size)
        target_nodes = target_nodes.argmax(dim=-1)
        target_edges = target_edges.argmax(dim=-1)
        # target_nodes = target_nodes.reshape(self.batch_size, -1)
        # target_nodes = torch.argmax(target_nodes, dim=-1).reshape(-1)
        # target_edges = torch.argmax(target_edges, dim=-1).reshape(-1)
        # target_edges = target_edges.reshape(self.batch_size, -1)
        # target = torch.cat((target_nodes, target_edges), dim=1)
        bce_loss_nodes = F.cross_entropy(pred_nodes, target_nodes, reduction='sum')
        bce_loss_edges = F.cross_entropy(pred_edges, target_edges, reduction='sum')
        bce_loss = bce_loss_nodes + bce_loss_edges
        # bce_loss = F.mse_loss(pred_edges, target)
        # kld_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        # loss = bce_loss + kld_loss
        loss = bce_loss
        self.log('bce_loss', bce_loss, on_epoch=True, prog_bar=True)
        # self.log('kld_loss', kld_loss, on_epoch=True, prog_bar=True)
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())
