from typing import List, Union, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.special import comb
from torch import Tensor
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Sequential, Tanh, Sigmoid
from torch_geometric.data import Batch

from modeling.utils import convert_to_dense


class Generator(torch.nn.Module):
    def __init__(self,
                 max_nodes_in_graph: int,
                 node_dim: int,
                 edge_dim: int,
                 latent_dim: int):
        super().__init__()

        def block(in_feat, out_feat, normalize=True) -> List[Union[Linear, BatchNorm1d, LeakyReLU]]:
            layers = [Linear(in_feat, out_feat)]
            if normalize:
                layers.append(BatchNorm1d(out_feat, 0.8))
            layers.append(LeakyReLU(0.2, inplace=True))
            return layers

        self.model = Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024)
        )
        self.node_nn = Sequential(
            Linear(1024, node_dim * max_nodes_in_graph),
            Tanh())
        self.edge_nn = Sequential(
            Linear(1024,
                   edge_dim * (comb(max_nodes_in_graph + 1, 2, exact=True) - max_nodes_in_graph)),
            Tanh())

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        combined_embedding = self.model(z)
        nodes = self.node_nn(combined_embedding)
        edges = self.edge_nn(combined_embedding)
        return nodes, edges


class Discriminator(torch.nn.Module):
    def __init__(self, max_nodes_in_graph: int, node_dim: int, edge_dim: int):
        super().__init__()
        in_dim = (comb(max_nodes_in_graph + 1, 2,
                       exact=True) - max_nodes_in_graph) * edge_dim + max_nodes_in_graph * node_dim
        self.model = Sequential(
            Linear(in_dim, 512),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 256),
            LeakyReLU(0.2, inplace=True),
            Linear(256, 1),
            Sigmoid(),
        )

    def forward(self, mol: Tensor) -> Tensor:
        validity = self.model(mol)
        return validity


class LitGAN(pl.LightningModule):

    def __init__(
            self,
            max_nodes_in_graph: int,
            node_dim: int,
            edge_dim: int,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 32,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(max_nodes_in_graph, node_dim, edge_dim, latent_dim)
        self.discriminator = Discriminator(max_nodes_in_graph, node_dim, edge_dim)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        return self.generator(z)

    def adversarial_loss(self, nodes: Tensor, edges: Tensor, targets: Tensor) -> Tensor:
        mol = torch.cat((nodes, edges), dim=1)
        preds = self.discriminator(mol)
        return F.binary_cross_entropy(preds, targets)

    def generator_loss(self, z: Tensor) -> Tensor:
        nodes, edges = self.generator(z)
        targets = torch.ones(self.hparams.batch_size, 1, device=self.device)
        return self.adversarial_loss(nodes, edges, targets)

    def discriminator_loss(self, real_nodes: Tensor, real_edges: Tensor, z: Tensor) -> Tensor:
        batch_size = self.hparams.batch_size
        real_targets = torch.ones(batch_size, 1, device=self.device)
        real_loss = self.adversarial_loss(real_nodes, real_edges, real_targets)

        fake_targets = torch.zeros(batch_size, 1, device=self.device)
        fake_nodes, fake_edges = self.generator(z)
        fake_loss = self.adversarial_loss(fake_nodes, fake_edges, fake_targets)

        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int) -> Tensor:
        batch_size = self.hparams.batch_size
        real_nodes, real_edges = convert_to_dense(batch,
                                                  self.hparams.max_nodes_in_graph,
                                                  batch_size)
        # Reshape for each use
        real_nodes = real_nodes.view(batch_size, -1)
        real_edges = real_edges.view(batch_size, -1)

        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)

        result = None
        if optimizer_idx == 0:
            result = self.generator_loss(z)
            self.log('g_loss', result, on_epoch=True, prog_bar=True)

        if optimizer_idx == 1:
            result = self.discriminator_loss(real_nodes, real_edges, z)
            self.log('d_loss', result, on_epoch=True, prog_bar=True)

        return result

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        real_nodes, real_edges = self.convert_to_dense(batch)
        z = torch.randn(self.hparams.batch_size, self.hparams.latent_dim, device=self.device)

        g_loss = self.generator_loss(z)
        d_loss = self.discriminator_loss(real_nodes, real_edges, z)
        total_loss = g_loss + d_loss
        self.log('val_loss', total_loss)

        return total_loss

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[None]]:
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
