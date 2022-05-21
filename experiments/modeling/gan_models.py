import numpy as np

from typing import List, Union, Tuple

import pytorch_lightning as pl
import torch
from rdkit.Chem.Draw import MolToImage
from scipy.special import comb
from torch import Tensor
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Sequential, Tanh, Sigmoid
from torch_geometric.data import Batch

from data_handling.utils import matrices2mol
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
        self.final_layer = Sequential(
            Linear(1024, node_dim * max_nodes_in_graph + edge_dim * (comb(max_nodes_in_graph + 1, 2, exact=True) - max_nodes_in_graph)))

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        embedding = self.model(z)
        mol = self.final_layer(embedding)
        return mol


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
            lambda_gp: float = 10.0,
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

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generator_loss(self, z: Tensor) -> Tensor:
        mols = self.generator(z)
        return -torch.mean(self.discriminator(mols))

    def discriminator_loss(self, real_nodes: Tensor, real_edges: Tensor, z: Tensor) -> Tensor:
        lambda_gp = self.hparams.lambda_gp
        real_mols = torch.cat((real_nodes, real_edges), dim=1)
        real_validity = self.discriminator(real_mols)

        fake_mols = self.generator(z)
        fake_validity = self.discriminator(fake_mols)

        gradient_penalty = self.compute_gradient_penalty(real_mols.data, fake_mols.data)

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
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

    def configure_optimizers(self):
        n_critic = 5

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic}
        )
