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

class PairwiseDistance(torch.nn.Module):  # (B, N,3), (B, M,3) -> (B, N, M)

    def __init__(self,
                 cut_dist_bohr=3.,
                 is_self_interacting=True,
                 epsilon: float = 10e-12):
        super().__init__()
        self.cut_dist_bohr = cut_dist_bohr
        self.min_dist = -10 if is_self_interacting == True else 10e-4
        self.default_type = torch.get_default_dtype()
        self.function = torch.vmap(self._non_batch_pairwise_distance)
        self.epsilon = epsilon

    def _non_batch_pairwise_distance(self, source, target):
        tmp_ = target[:, None] - source
        dist = torch.linalg.norm(tmp_, axis=-1, keepdim=True)
        mask = ((self.min_dist < dist) & (dist < self.cut_dist_bohr)).to(self.default_type)
        direction = tmp_ / (dist + self.epsilon)
        return dist, direction, mask

    def forward(self, source, target):
        dist, direction, mask = self.function(source, target)
        return dist, direction, mask