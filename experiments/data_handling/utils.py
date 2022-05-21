import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9

RAW_DATA_PATH: str = str(Path(Path(__file__).parent, 'raw_data').absolute())

atom_decoder = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}
bond_decoder = {1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE, 4: BT.AROMATIC}


class MetaDataset(NamedTuple):
    """
    Poorly named class to hold our datasplits, and meta information.
    """
    num_node_features: int
    train: Dataset
    val: Dataset
    test: Dataset


def matrices2mol(node_labels,
                 edge_labels,
                 strict=True):
    mol = Chem.RWMol()
    map = np.full(node_labels.shape[0], -1, dtype=int)
    for idx_r, node_label in enumerate(node_labels):
        if node_label != 0:
            map[idx_r] = mol.AddAtom(Chem.Atom(atom_decoder[node_label]))

    flattened_idx = 0
    for idx_r, node_label in enumerate(node_labels):
        for idx_c in range(idx_r):
            edge_label = edge_labels[flattened_idx]
            if node_label != 0 and edge_label != 0:
                if map[idx_r] != -1 and map[idx_c] != -1:
                    try:
                        mol.AddBond(map[idx_r].item(), map[idx_c].item(), bond_decoder[edge_label])
                    except Exception as e:
                        pass
            flattened_idx += 1
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(e)
            mol = None

    return mol


def load_data() -> MetaDataset:
    """
    Loads the dataset. Will take an arg in the future, for now loads the only dataset.
    Also purposely puts aside 40% of the data_handling as unseen, since we are just experimenting
    for now.
    :return: A train test split. Which is 60%, 20% and 20% of 40% of the data_handling. Poorly
    explained, but see below.
    """
    dataset = QM9(RAW_DATA_PATH)
    num_total_instances = len(dataset)
    all_data_for_now, _ = random_split(dataset, [math.floor(num_total_instances * .6),
                                                 math.ceil(num_total_instances * .4)])
    num_instances_for_now = len(all_data_for_now)
    train, val, test = random_split(all_data_for_now,
                                    [math.ceil(num_instances_for_now * .6),
                                     math.ceil(num_instances_for_now * .2),
                                     math.floor(num_instances_for_now * .2)])

    return MetaDataset(dataset.num_node_features, train, val, test)
