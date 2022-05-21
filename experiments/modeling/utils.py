from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj


def pad_a(a: np.ndarray, dim):
    shp = list(a.shape)
    shp[-1] = 1
    padding_indicators = np.zeros(shp)
    return np.concatenate([padding_indicators, a], axis=dim)


def pad_t(t: Tensor, dim):
    shp = list(t.shape)
    shp[-1] = 1
    padding_indicators = torch.zeros(shp)
    return torch.cat([padding_indicators, t], dim)


def convert_to_dense(
        batch: Batch, max_nodes_in_graph: int, batch_size: int) -> Tuple[Tensor, Tensor]:
    edge_index = batch['edge_index']
    edge_attr = batch['edge_attr']
    batch_mask = batch['batch']
    x = batch['x']

    x, real_data_indices = to_dense_batch(x, batch_mask,
                                          max_num_nodes=max_nodes_in_graph)
    # Add zeros to first index of node features so argmax(x) = 0 can indicate no atom present
    x = pad_t(x, 2)

    # Add zeroes to first index of edge features so argmax(adj) = 0 can indicate no edge present
    adj = to_dense_adj(edge_index, batch_mask, edge_attr,
                       max_num_nodes=max_nodes_in_graph)
    adj = pad_t(adj, 3)

    # Don't need a full matrix as edge connections are symmetric in this case
    indices = torch.tril_indices(max_nodes_in_graph, max_nodes_in_graph, -1)
    adj = adj[:, indices[0], indices[1]]

    return x, adj
