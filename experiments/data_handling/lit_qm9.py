import math
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Batch
from torch_geometric.datasets import QM9

RAW_DATA_PATH: str = str(Path(Path(__file__).parent, 'raw_data').absolute())


class LitQM9(LightningDataModule):
    def __init__(self, data_dir: str = RAW_DATA_PATH, batch_size: int = 32):
        super().__init__()
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size

    @property
    def max_nodes_per_mol(self) -> int:
        return 29

    @property
    def num_node_types(self) -> int:
        return 5

    @property
    def num_edge_types(self) -> int:
        return 4

    def prepare_data(self) -> None:
        QM9(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(self.data_dir)
        num_total_instances = len(dataset)
        all_data_for_now, _ = random_split(dataset, [math.floor(num_total_instances * .6),
                                                     math.ceil(num_total_instances * .4)])
        num_instances_for_now = len(all_data_for_now)
        self.train, self.val, self.test = random_split(all_data_for_now,
                                                       [math.ceil(num_instances_for_now * .6),
                                                        math.ceil(num_instances_for_now * .2),
                                                        math.floor(num_instances_for_now * .2)])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          num_workers=4,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val, num_workers=4, batch_size=self.batch_size, drop_last=True)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def on_before_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        # Last column is not node type
        batch['x'] = batch['x'][:, 0:5]
        return batch
