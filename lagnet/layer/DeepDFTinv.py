# MIT License
#
# Copyright (c) 2025 AIRI - Artificial Intelligence Research Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from ase.units import Bohr

from .general import PairwiseDistance
from ..third_party import gaussian_expansion, ShiftedSoftplus


class DeepDFTinv_expansion(torch.nn.Module):

    def __init__(self, cut_dist_bohr, features=20,
                 self_send=True, epsilon: float = 10e-12, distance_mul=Bohr):
        super().__init__()
        self.features = features
        self.cut_dist_bohr = cut_dist_bohr
        self.gaussian_expansion_step = cut_dist_bohr/features
        self.pw_dist = PairwiseDistance(cut_dist_bohr, is_self_interacting=self_send)
        self.epsilon = epsilon
        self.self_send = self_send
        self.distance_mul = distance_mul

    def forward(self, source, target):
        dist, direction, mask = self.pw_dist(source, target)

        expansion_2 = gaussian_expansion(dist*self.distance_mul + self.epsilon, [(0.0, self.gaussian_expansion_step, self.cut_dist_bohr)])

        return dist, direction, mask, expansion_2


def atom_features_to_edge_features(source, target):
    shape1 = source.shape[-3]
    shape2 = target.shape[-3]

    tensor_ext1 = source[..., None, :, :, :].repeat(1, shape2, 1, 1, 1)
    tensor_ext2 = target[..., None, :, :].repeat(1, 1, shape1, 1, 1)

    ret = torch.concatenate([tensor_ext1, tensor_ext2], axis=-1)

    return ret


class DeepDFTinv_Message_NoReciever(torch.nn.Module):

    def __init__(self, cutoff, size_of_expansion=20, num_features=32):
        super().__init__()
        self.cutoff = cutoff
        self.num_features = num_features

        self.msg_function_edge = torch.nn.Sequential(
            torch.nn.Linear(size_of_expansion, num_features),
            ShiftedSoftplus(),
            torch.nn.Linear(num_features, num_features),
        )
        self.msg_function_node = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            ShiftedSoftplus(),
            torch.nn.Linear(num_features, num_features),
        )

        self.soft_cutoff_func = lambda x: 1.0 - torch.sigmoid(
            5 * (x - (cutoff - 1.5))
        ) # В SchNet cosine. Это нарисовать и глазами посмотреть

    def forward(self, scalar, expansion, mask, edge_distance):

        gates = self.msg_function_edge(expansion) * self.soft_cutoff_func(
            edge_distance
        )
        nodes = self.msg_function_node(scalar)

        result = mask[..., None]*gates[...,None,:]*nodes[...,None,:,:,:]
        message_sum = torch.sum(result, axis=-3)

        return message_sum


class DeepDFTinv_Message_WithReciever(torch.nn.Module):

    def __init__(self, cutoff, size_of_expansion=20, num_features=32):
        super().__init__()
        self.cutoff = cutoff
        self.num_features = num_features

        self.msg_function_edge = torch.nn.Sequential(
            torch.nn.Linear(size_of_expansion, num_features),
            ShiftedSoftplus(),
            torch.nn.Linear(num_features, num_features),
        )
        self.msg_function_node = torch.nn.Sequential(
            torch.nn.Linear(2*num_features, num_features),
            ShiftedSoftplus(),
            torch.nn.Linear(num_features, num_features),
        )

        self.soft_cutoff_func = lambda x: 1.0 - torch.sigmoid(
            5 * (x - (cutoff - 1.5))
        )

    def forward(self, scalar, scalar_reciever, expansion, mask, edge_distance):

        gates = self.msg_function_edge(expansion) * self.soft_cutoff_func(
            edge_distance
        )
        pairwise_atom_features = atom_features_to_edge_features(scalar, scalar_reciever)
        nodes = self.msg_function_node(pairwise_atom_features)

        result = mask[..., None] * (gates[..., None, :] * nodes)
        message_sum = torch.sum(result, axis=-3)

        return message_sum


class DeepDFTinv_Model(torch.nn.Module):

    def __init__(self, cutoff,
                 depth=2,
                 size_of_expansion=20,
                 num_features=32,
                 atom_embedding_size=36):
        super().__init__()
        self.cutoff = cutoff
        self.size_of_expansion = size_of_expansion
        self.num_features = num_features
        self.depth = depth
        self.embedding = torch.nn.Embedding(num_embeddings=atom_embedding_size,
                                            embedding_dim=self.num_features)

        self.interaction_between_atoms = torch.nn.ModuleList([DeepDFTinv_Message_WithReciever(cutoff,
                                                                                              size_of_expansion=size_of_expansion,
                                                                                              num_features=num_features) for i in range(depth)])

        self.state_transition_function = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(num_features, num_features),
                                                                                  ShiftedSoftplus(),
                                                                                  torch.nn.Linear(num_features, num_features))
                                                              for i in range(depth)])

        self.probe_state_gate_functions = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(num_features, num_features),
                    ShiftedSoftplus(),
                    torch.nn.Linear(num_features, num_features),
                    torch.nn.Sigmoid(),
                )
                for _ in range(depth)
            ]
        )

        self.probe_state_transition_functions = torch.nn.ModuleList(
            [
                    torch.nn.Sequential(
                    torch.nn.Linear(num_features, num_features),
                    ShiftedSoftplus(),
                    torch.nn.Linear(num_features, num_features),
                )
                for _ in range(depth)
            ]
        )

        self.interaction_atom_to_probe = torch.nn.ModuleList([DeepDFTinv_Message_WithReciever(cutoff,
                                                                                              size_of_expansion=size_of_expansion,
                                                                                              num_features=num_features) for i in range(depth)])

        self.rbf_expansion = DeepDFTinv_expansion(cutoff, features=size_of_expansion, self_send=False, distance_mul=Bohr)

    def forward(self, atom_coords, probe_coords, atom_numbers):
        scalar = self.embedding(atom_numbers.to(torch.int))[..., None, :]
        scalar_probes = torch.zeros((atom_coords.shape[0], probe_coords.shape[1], 1, self.num_features), device=atom_coords.device)

        dist, direction, mask, expansion = self.rbf_expansion(atom_coords, atom_coords)
        dist_p1, direction_p1, mask_p1, expansion_p1 = self.rbf_expansion(atom_coords, probe_coords)

        for i in range(self.depth):
            delta_scalar = self.interaction_between_atoms[i](scalar, scalar, expansion, mask, dist)
            scalar = scalar + self.state_transition_function[i](delta_scalar)

            delta_scalar_probes = self.interaction_atom_to_probe[i](scalar, scalar_probes, expansion_p1, mask_p1, dist_p1)
            gates = self.probe_state_gate_functions[i](scalar_probes)
            scalar_probes = scalar_probes * gates + (1 - gates) * self.probe_state_transition_functions[i](delta_scalar_probes)

        return scalar_probes

