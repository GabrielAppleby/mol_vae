from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
from pytorch_lightning import Callback, LightningModule, Trainer
from rdkit.Chem.Draw import MolToImage
from scipy.special import comb
from torchvision.transforms import ToTensor

from data_handling.utils import matrices2mol
from modeling.utils import pad_a, convert_to_dense


class GANImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    """

    def __init__(
            self,
            max_nodes_per_mol: int,
            num_node_types: int,
            num_edge_types: int,
            num_samples: int = 3,
            nrow: int = 8,
            padding: int = 2
    ) -> None:
        """

        """
        super().__init__()
        self.max_nodes_per_mol = max_nodes_per_mol
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dim = (self.num_samples, pl_module.hparams.latent_dim)  # type: ignore[union-attr]
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            mols = pl_module(z).detach().numpy()
            pl_module.train()

        nodes, edges = np.split(mols, [self.max_nodes_per_mol * self.num_node_types], axis=1)
        nodes = nodes.reshape(self.num_samples, self.max_nodes_per_mol, self.num_node_types)
        edges = edges.reshape(
            self.num_samples,
            (comb(self.max_nodes_per_mol + 1, 2, exact=True) - self.max_nodes_per_mol),
            self.num_edge_types)

        edges = np.argmax(edges, axis=-1)
        nodes = np.argmax(nodes, axis=-1)
        mols = []
        for i in range(nodes.shape[0]):
            mol = matrices2mol(nodes[i], edges[i])
            if mol != None:
                mol.append(MolToImage(mol))

        if len(mols) > 0:
            grid = torchvision.utils.make_grid(
                tensor=mols,
                nrow=self.nrow,
                padding=self.padding
            )
            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)


class TrainImageReconstructionLogger(Callback):
    def __init__(
        self,
        max_nodes_per_mol: int,
        num_node_types: int,
        num_edge_types: int,
        nrow: int = 8,
        padding: int = 2
    ) -> None:
        super().__init__()
        self.max_nodes_per_mol = max_nodes_per_mol
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.nrow = nrow
        self.padding = padding
        self.to_tensor = ToTensor()

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_loader = pl_module.train_dataloader()
        batch_size = train_loader.batch_size
        train_sample = next(iter(train_loader)).to(pl_module.device)
        train_sample.x = train_sample.x[:, 0:5]
        # generate images
        with torch.no_grad():
            pl_module.eval()
            test = pl_module(train_sample.x, train_sample.edge_index, train_sample.edge_attr, train_sample.batch).detach().numpy()
            pl_module.train()


        # Real
        target_nodes, target_edges = convert_to_dense(
            train_sample, self.max_nodes_per_mol, batch_size)
        target_nodes = torch.argmax(target_nodes, dim=-1).detach().numpy()
        target_edges = torch.argmax(target_edges, dim=-1).detach().numpy()

        mols = []
        for i in range(target_nodes.shape[0]):
            mol = matrices2mol(target_nodes[i], target_edges[i])
            if mol != None:
                mols.append(MolToImage(mol))
        mols = [self.to_tensor(mol) for mol in mols]
        if len(mols) > 0:
            grid = torchvision.utils.make_grid(
                tensor=mols,
                nrow=self.nrow,
                padding=self.padding
            )
            str_title = f"{pl_module.__class__.__name__}_actual_images"
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

        # Pred
        nodes, edges = np.split(test, [self.max_nodes_per_mol * self.num_node_types], axis=1)
        # print(self.max_nodes_per_mol * self.num_node_types)
        # nodes, edges = np.split(mols, [self.max_nodes_per_mol * self.num_node_types], axis=1)
        nodes = nodes.reshape(batch_size, self.max_nodes_per_mol, self.num_node_types)
        edges = edges.reshape(
            batch_size,
            (comb(self.max_nodes_per_mol + 1, 2, exact=True) - self.max_nodes_per_mol),
            self.num_edge_types)

        nodes = np.argmax(nodes, axis=-1)
        edges = np.argmax(edges, axis=-1)
        mols = []
        for i in range(nodes.shape[0]):
            mol = matrices2mol(nodes[i], edges[i])
            if mol != None:
                mols.append(MolToImage(mol))
        mols = [self.to_tensor(mol) for mol in mols]
        if len(mols) > 0:
            grid = torchvision.utils.make_grid(
                tensor=mols,
                nrow=self.nrow,
                padding=self.padding
            )
            str_title = f"{pl_module.__class__.__name__}_pred_images"
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)
