import pathlib

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from data_handling.lit_qm9 import LitQM9
from modeling.callbacks import GANImageSampler, TrainImageReconstructionLogger
from modeling.gan_models import LitGAN
from modeling.vae_models import LitAutoEncoder

CURRENT_DIR = pathlib.Path(__file__).parent
SAVED_MODEL_PATH = pathlib.Path(CURRENT_DIR, 'basic_gan.pt')
SAVED_EMBEDDINGS_PATH = pathlib.Path(CURRENT_DIR, 'gcn_embeddings.npz')

RANDOM_SEED: int = 42


def set_random_seeds(random_seed: int) -> None:
    """
    Set the random seed for any libraries used.
    :param random_seed: The random seed to set.
    :return: None.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def main():
    set_random_seeds(RANDOM_SEED)

    batch_size = 32

    dm: LitQM9 = LitQM9(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    max_nodes_per_mol = dm.max_nodes_per_mol
    num_node_types_gan = dm.num_node_types + 1
    num_node_types_vae = dm.num_node_types
    num_edge_types_gan = dm.num_edge_types + 1
    num_edge_types_vae = dm.num_edge_types

    trainer: Trainer = Trainer(
        gpus=0,
        max_epochs=500,
        # callbacks=[TrainImageReconstructionLogger(max_nodes_per_mol,
        #                                           num_node_types_gan,
        #                                           num_edge_types_gan),
        #            EarlyStopping(monitor='val_loss', patience=10)])
        callbacks=[GANImageSampler(max_nodes_per_mol,
                                   num_node_types_gan,
                                   num_edge_types_gan,
                                   num_samples=8)])
    model: LitGAN = LitGAN(max_nodes_per_mol, num_node_types_gan, num_edge_types_gan, batch_size=batch_size)
    # model: LitAutoEncoder = LitAutoEncoder(
    #     max_nodes_per_mol, num_node_types_vae, num_edge_types_vae, 64, 64, batch_size=batch_size)
    trainer.fit(model, datamodule=dm)
    # torch.save(model, SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()
