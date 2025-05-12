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
from ..third_party import sinc_expansion, cosine_cutoff


class DeepDFTeqv_expansion(torch.nn.Module):

    def __init__(self, cut_dist_bohr, features=20,
                 self_send=True, epsilon: float = 10e-12, distance_mul=Bohr):
        super().__init__()
        self.features = features
        self.cut_dist_bohr = cut_dist_bohr
        self.pw_dist = PairwiseDistance(cut_dist_bohr, is_self_interacting=self_send)
        self.epsilon = epsilon
        self.self_send = self_send
        self.distance_mul = distance_mul

    def forward(self, source, target):
        dist, direction, mask = self.pw_dist(source, target)

        expansion_2 = sinc_expansion(dist*self.distance_mul + self.epsilon, [(self.features, self.cut_dist_bohr)])

        return dist, direction, mask, expansion_2

class DeepDFTeqv_Update(torch.nn.Module):

    def __init__(self, num_features=32):
        self.num_features = num_features
        super().__init__()
        self.upd_right = torch.nn.Sequential(torch.nn.Linear(2 * num_features, num_features),
                                             torch.nn.SiLU(),
                                             torch.nn.Linear(num_features, 3 * num_features))

        self.V = torch.nn.Linear(num_features, num_features, bias=False)
        self.U = torch.nn.Linear(num_features, num_features, bias=False)

    def forward(self, scalar, vector):
        left = self.U(vector)
        right = self.V(vector)
        right_norm = torch.linalg.norm(right, axis=-2, keepdim=True)

        right_concat = torch.cat((scalar, right_norm), axis=-1)

        right_features = self.upd_right(right_concat)

        a, b, c = torch.split(right_features, (self.num_features, self.num_features, self.num_features), dim=-1)

        delta_v = a * left
        inner_prod = torch.sum(left * right, dim=-2, keepdim=True)
        delta_s = inner_prod * b + c

        return delta_s, delta_v

class DeepDFTeqv_Message(torch.nn.Module):

    def __init__(self, cutoff, size_of_expansion=20, num_features=32):
        super().__init__()
        self.cutoff = cutoff
        self.num_features = num_features
        self.msg_left = torch.nn.Sequential(torch.nn.Linear(num_features, num_features),
                                            torch.nn.SiLU(),
                                            torch.nn.Linear(num_features, 3 * num_features))

        self.msg_right = torch.nn.Sequential(torch.nn.Linear(size_of_expansion, 3 * num_features))

    def forward(self, s, v, expansion, direction, mask, edge_distance):
        left_mul = self.msg_left(s)
        right_mul_preliminary = self.msg_right(expansion)[..., None, :]
        f_cut_mul = cosine_cutoff(edge_distance, self.cutoff)[..., None]
        right_mul = right_mul_preliminary * f_cut_mul
        a, b, c = torch.split(left_mul * right_mul, (self.num_features, self.num_features, self.num_features), dim=-1)

        mask_ = mask[..., None]
        delta_s = torch.sum(mask_ * b, axis=-3)
        delta_v_right = direction[..., None] * c
        delta_v_left = v * a
        delta_v = (mask_ * (delta_v_left + delta_v_right)).sum(axis=-3)

        return delta_s, delta_v


class DeepDFTeqv_Model(torch.nn.Module):

    def __init__(self,
                 cutoff=15,
                 depth=3,
                 size_of_expansion=20,
                 num_features=32,
                 atom_embedding_size=36):
        super().__init__()
        self.cutoff = cutoff
        self.depth = depth
        self.size_of_expansion = size_of_expansion
        self.num_features = num_features
        self.atom_embedding_size = atom_embedding_size
        self.embedding = torch.nn.Embedding(num_embeddings=atom_embedding_size,
                                            embedding_dim=self.num_features)

        self.list_msg = torch.nn.ModuleList([DeepDFTeqv_Message(cutoff,
                                                                size_of_expansion=size_of_expansion,
                                                                num_features=num_features) for i in range(depth)])
        self.list_upd = torch.nn.ModuleList([DeepDFTeqv_Update(num_features=num_features) for i in range(depth)])

        self.list_msg_p = torch.nn.ModuleList([DeepDFTeqv_Message(cutoff,
                                                                  size_of_expansion=size_of_expansion,
                                                                  num_features=num_features) for i in range(depth)])
        self.list_upd_p = torch.nn.ModuleList([DeepDFTeqv_Update(num_features=num_features) for i in range(depth)])

        self.rbf_expansion = DeepDFTeqv_expansion(cutoff, features=size_of_expansion, self_send=False, distance_mul=Bohr)
        self.rbf_expansion_p = DeepDFTeqv_expansion(cutoff, features=size_of_expansion, self_send=False, distance_mul=Bohr)

    def forward(self, atom_coords, probe_coords, atom_numbers):
        scalar = self.embedding(atom_numbers.to(torch.int))[..., None, :]
        vector = torch.zeros((atom_coords.shape[0], atom_coords.shape[1], 3, self.num_features),
                             device=atom_coords.device)

        dist, direction, mask, expansion = self.rbf_expansion(atom_coords, atom_coords)
        dist_p1, direction_p1, mask_p1, expansion_p1 = self.rbf_expansion_p(atom_coords, probe_coords)

        p1_scalar = torch.zeros((probe_coords.shape[0], probe_coords.shape[1], 1, self.num_features), device=atom_coords.device)
        p1_vector = torch.zeros((probe_coords.shape[0], probe_coords.shape[1], 3, self.num_features), device=atom_coords.device)

        for ind_d in range(self.depth):
            # PaiNN message
            delta_scalar, delta_vector = self.list_msg[ind_d](scalar, vector, expansion, direction, mask, dist)
            scalar = scalar + delta_scalar
            vector = vector + delta_vector

            delta_scalar, delta_vector = self.list_upd[ind_d](scalar, vector)
            scalar = scalar + delta_scalar
            vector = vector + delta_vector

            # DeepDFT update
            delta_p1_scalar, delta_p1_vector = self.list_msg_p[ind_d](scalar,
                                                                     vector,
                                                                     expansion_p1,
                                                                     direction_p1,
                                                                     mask_p1,
                                                                     dist_p1)
            p1_scalar = p1_scalar + delta_p1_scalar
            p1_vector = p1_vector + delta_p1_vector

            delta_p1_scalar, delta_p1_vector = self.list_upd_p[ind_d](p1_scalar, p1_vector)
            p1_scalar = p1_scalar + delta_p1_scalar
            p1_vector = p1_vector + delta_p1_vector

        return p1_scalar, p1_vector


