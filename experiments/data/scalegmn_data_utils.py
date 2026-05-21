from typing import Dict

import torch
import torch_geometric


def to_pyg_batch(node_features,
                 edge_features,
                 edge_index,
                 node2type=None,
                 edge2type=None,
                 direction='forward',
                 label=None,
                 hidden_nodes=None,
                 first_layer_nodes=None
                 ):
    if direction in ['forward', 'backward']:
        edge_features = edge_features if direction == 'forward' else edge_features.transpose(-2, -3)
        edge_index = edge_index if direction == 'forward' else torch.flip(edge_index, [0])
        data = torch_geometric.data.Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features[edge_index[0], edge_index[1]],
                node2type=node2type if node2type is not None else None,
                edge2type=edge2type if edge2type is not None else None,
                label=label,
                mask_hidden=hidden_nodes,
                mask_first_layer=first_layer_nodes
            )
        return data, None

    elif direction == 'bidirectional':
        data = torch_geometric.data.Data(
            x=node_features,
            edge_index=edge_index,
            bw_edge_index=torch.flip(edge_index, [0]),
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            node2type=node2type if node2type is not None else None,
            edge2type=edge2type if edge2type is not None else None,
            label=label,
            mask_hidden=hidden_nodes,
            mask_first_layer=first_layer_nodes
        )
        return data, None


def get_node_types(nodes_per_layer):
    node_types = []
    type = 0
    for i, el in enumerate(nodes_per_layer):
        if i == 0:  # first layer
            for _ in range(el):
                node_types.append(type)
                type += 1
        elif i > 0 and i < len(nodes_per_layer) - 1:  #  hidden layers
            for _ in range(el):
                node_types.append(type)
            type += 1
        elif i == len(nodes_per_layer) - 1:  # last layer
            for _ in range(el):
                node_types.append(type)
                type += 1
    return torch.tensor(node_types)


def get_edge_types(nodes_per_layer):
    edge_types = []
    type = 0
    for i, el in enumerate(nodes_per_layer[:-1]):
        if i == 0:  # first layer
            for _ in range(el):
                for neighbour in range(nodes_per_layer[i+1]):
                    edge_types.append(type)
                type += 1
        elif i > 0 and i < len(nodes_per_layer) - 2:  #  hidden layers
            for _ in range(el):
                for neighbour in range(nodes_per_layer[i+1]):
                    edge_types.append(type)
            type += 1
        elif i == len(nodes_per_layer) - 2:  # last layer
            for neighbour in range(nodes_per_layer[i + 1]):
                for _ in range(el):
                    edge_types.append(type)
                type += 1

    # from collections import Counter
    # print(Counter(edge_types))
    return torch.tensor(edge_types)


def nn_to_edge_index(layer_layout, device, dtype=torch.long):
    edge_index = []

    node_offset = 0
    nodes_per_layer = []
    for n in layer_layout:
        nodes_per_layer.append(list(range(node_offset, node_offset + n)))
        node_offset += n

    for i in range(1, len(layer_layout)):
        for j in nodes_per_layer[i - 1]:
            for k in nodes_per_layer[i]:
                edge_index.append([j, k])

    return torch.tensor(edge_index, device=device, dtype=dtype).T


def mask_input(conf: Dict) -> bool:
    return_mask = conf['scalegmn_args']['gnn_args']['msg_num_mlps'] == 3 \
                           or conf['scalegmn_args']['gnn_args']['upd_num_mlps'] == 3
    return return_mask


def mask_hidden(conf: Dict) -> bool:
    return_mask = conf['scalegmn_args']['gnn_args']['msg_equiv_on_hidden'] \
                    or conf['scalegmn_args']['gnn_args']['upd_equiv_on_hidden'] \
                    or conf['scalegmn_args']['gnn_args']['layer_msg_equiv_on_hidden'] \
                    or conf['scalegmn_args']['gnn_args']['layer_upd_equiv_on_hidden']
    return return_mask


class GraphBatcher:
    def __init__(
        self,
        layer_layout,
        direction='forward',
        node_pos_embed=False,
        edge_pos_embed=False,
        equiv_on_hidden=False,
        get_first_layer_mask=False
    ):
        self.layer_layout = layer_layout
        self.direction = direction
        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask

        self.edge_index = nn_to_edge_index(self.layer_layout, "cpu", dtype=torch.long)

        if self.node_pos_embed:
            self.node2type = get_node_types(self.layer_layout)
        if self.edge_pos_embed:
            self.edge2type = get_edge_types(self.layer_layout)

        # Since the current datasets have the same architecture for every datapoint, we can
        # create the below masks on initialization, rather than on __getitem__.
        if self.equiv_on_hidden:
            self.hidden_nodes = self.mark_hidden_nodes()
        if self.get_first_layer_mask:
            self.first_layer_nodes = self.mark_input_nodes()

    def mark_hidden_nodes(self) -> torch.Tensor:
        hidden_nodes = torch.tensor(
                [False for _ in range(self.layer_layout[0])] +
                [True for _ in range(sum(self.layer_layout[1:-1]))] +
                [False for _ in range(self.layer_layout[-1])]).unsqueeze(-1)
        return hidden_nodes

    def mark_input_nodes(self) -> torch.Tensor:
        input_nodes = torch.tensor(
            [True for _ in range(self.layer_layout[0])] +
            [False for _ in range(sum(self.layer_layout[1:]))]).unsqueeze(-1)
        return input_nodes

    @classmethod
    def batch_to_graphs(
        cls,
        weights,
        biases,
        input_emb=None,
        **kwargs
    ):
        num_nodes = weights[0].shape[0] + sum(w.shape[1] for w in weights)

        node_features = torch.zeros(num_nodes, biases[0].shape[-1])
        edge_features = torch.zeros(num_nodes, num_nodes, weights[0].shape[-1])

        row_offset = 0
        col_offset = weights[0].shape[0]  # no edge to input nodes
        for i, w in enumerate(weights):
            num_in, num_out, _ = w.shape
            edge_features[row_offset : row_offset + num_in, col_offset : col_offset + num_out] = w
            row_offset += num_in
            col_offset += num_out

        row_offset = weights[0].shape[0]  # no bias in input nodes

        if input_emb is not None:
            node_features[:, 0: row_offset] = input_emb
        else:
            node_features[:, 0: row_offset] = torch.tensor([1])  # set input node state to 1.

        for i, b in enumerate(biases):
            num_out, _ = b.shape
            node_features[row_offset : row_offset + num_out] = b
            row_offset += num_out

        return node_features, edge_features

    def get_graph_batch(self, weights, biases, label=None):
        node_features, edge_features = self.batch_to_graphs(weights, biases)
        batch, _ = to_pyg_batch(
            node_features,
            edge_features,
            self.edge_index,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            direction=self.direction,
            label=label,
            hidden_nodes=self.hidden_nodes if self.equiv_on_hidden else None,
            first_layer_nodes=self.first_layer_nodes if self.get_first_layer_mask else None
        )

        return batch


def get_batch_from_wb(weights, biases, batcher, label=None):

    class DummyBatch(torch.utils.data.Dataset):
        def __init__(self, weights, biases, batcher, label=None):
            self.weights = weights
            self.biases = biases
            self.batcher = batcher
            self.label = label

        def __len__(self):
            return len(self.weights[0])

        def __getitem__(self, index):
            return self.batcher.get_graph_batch(
                [w[index] for w in self.weights],
                [b[index] for b in self.biases],
                label=self.label[index] if self.label is not None else None
            )

    data = DummyBatch(weights, biases, batcher, label=label)
    bs = len(data)
    loader = torch_geometric.loader.DataLoader(
        dataset=data,
        batch_size=bs,
        shuffle=False,
    )
    batch = next(iter(loader))
    return batch
