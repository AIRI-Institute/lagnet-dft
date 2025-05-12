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
from e3nn import math as o3_math
from .general import PairwiseDistance


class DeepDFText0_expansion(torch.nn.Module):

    def __init__(self, cut_dist_bohr, min_value=-0.5, max_value=3.0, features=20, basis='bessel', cutoff=True,
                 self_send=True, epsilon: float = 10e-12):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.features = features
        self.basis = basis
        self.cutoff = cutoff
        self.pw_dist = PairwiseDistance(cut_dist_bohr, is_self_interacting=self_send)
        self.epsilon = epsilon
        self.self_send = self_send

    def forward(self, source, target):
        dist, direction, mask = self.pw_dist(source, target)

        expansion_2 = o3_math.soft_one_hot_linspace(dist[..., 0] + self.epsilon,
                                                    self.min_value,
                                                    self.max_value,
                                                    self.features,
                                                    basis=self.basis,
                                                    cutoff=self.cutoff)

        return dist, direction, mask, expansion_2


class DeepDFText0_Update(torch.nn.Module):

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


class DeepDFText0_UpdateExtended(torch.nn.Module):

    def __init__(self,
                 inout_features=32):
        super().__init__()

        self.inout_features = inout_features

        self.update_mixing = torch.nn.Sequential(
            torch.nn.Linear(12*inout_features, 8*inout_features),
            torch.nn.SiLU(),
            torch.nn.Linear(8 * inout_features, 16*inout_features))

        self.A = torch.nn.Linear(inout_features, inout_features, bias=False)
        self.B = torch.nn.Linear(inout_features, inout_features, bias=False)

    def forward(self, scalar_state, vector_state, scalar_upd, vector_upd):
        vector_state2 = self.A(vector_state)
        vector_upd2 = self.B(vector_upd)

        norm0 = torch.linalg.norm(vector_state, axis=-2, keepdim=True)
        norm1 = torch.linalg.norm(vector_upd, axis=-2, keepdim=True)
        norm2 = torch.linalg.norm(vector_state2, axis=-2, keepdim=True)
        norm3 = torch.linalg.norm(vector_upd2, axis=-2, keepdim=True)

        cross0 = torch.cross(vector_state, vector_upd, dim=-2)
        cross2 = torch.cross(vector_state2, vector_upd, dim=-2)

        cross0_norm = torch.linalg.norm(cross0, axis=-2, keepdim=True)**2
        cross2_norm = torch.linalg.norm(cross2, axis=-2, keepdim=True)**2

        mul0 = vector_state*vector_upd
        mul1 = vector_state*vector_upd2
        mul2 = vector_state2*vector_upd
        mul3 = vector_state2*vector_upd2

        mul0_norm = torch.sum(mul0, dim=-2, keepdim=True)
        mul1_norm = torch.sum(mul1, dim=-2, keepdim=True)
        mul2_norm = torch.sum(mul2, dim=-2, keepdim=True)
        mul3_norm = torch.sum(mul3, dim=-2, keepdim=True)

        features = torch.cat((norm0, norm1, norm2, norm3,
                                     cross0_norm, cross2_norm,
                                     mul0_norm, mul1_norm, mul2_norm, mul3_norm,
                                     scalar_state, scalar_upd), axis=-1)

        gates = self.update_mixing(features)


        gate_vector_state2, gate_vector_upd, gate_vector_upd2,\
        gate_scalar_upd,\
        gate_norm0, gate_norm1, gate_norm2, gate_norm3, \
        gate_cross0_norm, gate_cross1_norm, gate_cross2_norm, gate_cross3_norm, \
        gate_mul0_norm, gate_mul1_norm, gate_mul2_norm, gate_mul3_norm = torch.split(gates, [self.inout_features]*16, dim=-1)

        new_vector = vector_state + gate_vector_state2*vector_state2 + \
                     gate_vector_upd*vector_upd + gate_vector_upd2*vector_upd2

        new_scalar = scalar_state + gate_scalar_upd * scalar_upd + \
                     gate_norm0*norm0 + gate_norm1*norm1 + gate_norm2*norm2 + gate_norm3*norm3 + \
                     gate_cross0_norm*cross0_norm  + gate_cross2_norm*cross2_norm + \
                     gate_mul0_norm * mul0_norm + gate_mul1_norm * mul1_norm + \
                     gate_mul2_norm * mul2_norm + gate_mul3_norm * mul3_norm

        return new_scalar, new_vector


class DeepDFText0_Message(torch.nn.Module):

    def __init__(self, size_of_expansion=20, num_features=32):
        super().__init__()
        self.num_features = num_features
        self.msg_left = torch.nn.Sequential(torch.nn.Linear(num_features, num_features),
                                            torch.nn.SiLU(),
                                            torch.nn.Linear(num_features, 3 * num_features))

        self.msg_right = torch.nn.Sequential(torch.nn.Linear(size_of_expansion, 3 * num_features))

    def forward(self, s, v, expansion, direction, mask):
        left_mul = self.msg_left(s)
        right_mul = self.msg_right(expansion)[..., None, :]
        a, b, c = torch.split(left_mul * right_mul, (self.num_features, self.num_features, self.num_features), dim=-1)

        mask_ = mask[..., None]
        delta_s = torch.sum(mask_ * b, axis=-3)
        delta_v_right = direction[..., None] * c
        delta_v_left = v * a
        delta_v = torch.sum(mask_ * (delta_v_left + delta_v_right), axis=-3)

        return delta_s, delta_v

class DeepDFText0_MessageAndUpdate(torch.nn.Module):

    def __init__(self, num_features=32, size_of_expansion=20):
        super().__init__()
        self.message = DeepDFText0_Message(num_features = num_features, size_of_expansion=size_of_expansion)
        self.update = DeepDFText0_Update(num_features = num_features)


    def forward(self, scalar, vector, expansion, direction, mask):
        message_delta, vector_delta = self.message(scalar, vector, expansion, direction, mask)
        scalar = scalar + message_delta
        vector = vector + vector_delta

        message_delta, vector_delta = self.update(scalar, vector)
        scalar = scalar + message_delta
        vector = vector + vector_delta

        return scalar, vector

    def forward_init(self, scalar, vector, expansion, direction, mask):
        scalar_init, vector_init = self.message(scalar, vector, expansion, direction, mask)
        scalar_delta, vector_delta = self.update(scalar_init, vector_init)
        scalar_init = scalar_init + scalar_delta
        vector_init = vector_init + vector_delta

        return scalar_init, vector_init

    def forward_mixing(self, scalar_source, vector_source,
                             scalar_target, vector_target,
                             expansion, direction, mask):

        scalar_delta, vector_delta = self.message(scalar_source, vector_source, expansion, direction, mask)
        scalar_target = scalar_target + scalar_delta
        vector_target = vector_target + vector_delta

        scalar_delta, vector_delta = self.update(scalar_target, vector_target)
        scalar_target = scalar_target + scalar_delta
        vector_target = vector_target + vector_delta

        return scalar_target, vector_target


class DeepDFText0_MessageAndUpdateExtended(torch.nn.Module):

    def __init__(self, num_features=32, size_of_expansion=20):
        super().__init__()
        self.message = DeepDFText0_Message(num_features=num_features, size_of_expansion=size_of_expansion)
        self.update = DeepDFText0_UpdateExtended(inout_features=num_features)

    def forward(self, scalar, vector, expansion, direction, mask):
        scalar_delta, vector_delta = self.message(scalar, vector, expansion, direction, mask)
        scalar_delta, vector_delta = self.update(scalar, vector, scalar_delta, vector_delta)
        return scalar_delta, vector_delta

    def forward_init(self, scalar, vector, expansion, direction, mask):
        scalar_delta, vector_delta = self.message(scalar, vector, expansion, direction, mask)
        scalar_init = torch.zeros_like(scalar_delta)
        vector_init = torch.zeros_like(vector_delta)

        scalar_delta, vector_delta = self.update(scalar_init, vector_init, scalar_delta, vector_delta)
        return scalar_delta, vector_delta

    def forward_mixing(self, scalar_source, vector_source,
                             scalar_target, vector_target,
                             expansion, direction, mask):

        scalar_delta, vector_delta = self.message(scalar_source, vector_source, expansion, direction, mask)
        scalar_delta, vector_delta = self.update(scalar_target, vector_target, scalar_delta, vector_delta)

        return scalar_delta, vector_delta


class DeepDFText0_Model(torch.nn.Module):

    def __init__(self,
                 depth=3,
                 size_of_expansion=20,
                 num_features=32,
                 atom_embedding_size=36,
                 painn_column_cutoff=3.,
                 painn_column_min_value=0.,
                 painn_column_max_value=3.,
                 painn_column_basis='bessel',
                 deepdft_column0_cutoff=3.,
                 deepdft_column0_min_value=-0.5,
                 deepdft_column0_max_value=3.0,
                 deepdft_column0_basis='bessel',
                 deepdft_column1_cutoff=20.,
                 deepdft_column1_min_value=0.0,
                 deepdft_column1_max_value=20.,
                 deepdft_column1_basis='fourier',
                 ):
        super().__init__()
        self.depth = depth
        self.size_of_expansion = size_of_expansion
        self.num_features = num_features
        self.atom_embedding_size = atom_embedding_size
        self.embedding = torch.nn.Embedding(num_embeddings=atom_embedding_size,
                                            embedding_dim=self.num_features)

        self.painn_column = torch.nn.ModuleList([DeepDFText0_MessageAndUpdate(size_of_expansion=size_of_expansion,
                                                                              num_features=num_features) for i in range(depth)])
        self.deepdft_column = torch.nn.ModuleList([DeepDFText0_MessageAndUpdateExtended(size_of_expansion=size_of_expansion,
                                                                                        num_features=num_features) for i in
                                                   range(depth + 1)])
        self.deepdft_column2 = torch.nn.ModuleList([DeepDFText0_MessageAndUpdateExtended(size_of_expansion=size_of_expansion,
                                                                                         num_features=num_features) for i in
                                                    range(depth + 1)])

        self.rbf_expansion = DeepDFText0_expansion(painn_column_cutoff,
                                                   min_value=painn_column_min_value,
                                                   max_value=painn_column_max_value,
                                                   basis=painn_column_basis)
        self.rbf_expansion_p = DeepDFText0_expansion(deepdft_column0_cutoff,
                                                     min_value=deepdft_column0_min_value,
                                                     max_value=deepdft_column0_max_value,
                                                     basis=deepdft_column0_basis)
        self.rbf_expansion_p2 = DeepDFText0_expansion(deepdft_column1_cutoff,
                                                      min_value=deepdft_column1_min_value,
                                                      max_value=deepdft_column1_max_value,
                                                      basis=deepdft_column1_basis)

    def forward(self, atom_coords, probe_coords, atom_numbers):
        scalar = self.embedding(atom_numbers.to(torch.int))[..., None, :]
        vector = torch.zeros((atom_coords.shape[0], atom_coords.shape[1], 3, self.num_features),
                             device=atom_coords.device)

        dist, direction, mask, expansion = self.rbf_expansion(atom_coords, atom_coords)
        dist_p1, direction_p1, mask_p1, expansion_p1 = self.rbf_expansion_p(atom_coords, probe_coords)
        dist_p2, direction_p2, mask_p2, expansion_p2 = self.rbf_expansion_p2(atom_coords, probe_coords)

        p1_scalar, p1_vector = self.deepdft_column[0].forward_init(scalar, vector,
                                                                     expansion_p1, direction_p1, mask_p1)
        p2_scalar, p2_vector = self.deepdft_column2[0].forward_init(scalar, vector,
                                                                      expansion_p2, direction_p2, mask_p2)

        for ind_d in range(self.depth):
            scalar, vector = self.painn_column[ind_d](scalar, vector, expansion, direction, mask)

            p1_scalar, p1_vector = self.deepdft_column[ind_d + 1].forward_mixing(scalar,
                                                                                 vector,
                                                                                 p1_scalar,
                                                                                 p1_vector,
                                                                                 expansion_p1,
                                                                                 direction_p1,
                                                                                 mask_p1)
            p2_scalar, p2_vector = self.deepdft_column2[ind_d + 1].forward_mixing(scalar,
                                                                                  vector,
                                                                                  p2_scalar,
                                                                                  p2_vector,
                                                                                  expansion_p2,
                                                                                  direction_p2,
                                                                                  mask_p2)

        return scalar, vector, p1_scalar, p1_vector, p2_scalar, p2_vector