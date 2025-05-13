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
import numpy as np
import pyscf
from pyscf import dft, gto
from sklearn.metrics.pairwise import pairwise_distances

N_TEST = 10
NON_DEGENERACY_TOLERANCE = 0.1

def gen_rotate_matrix():
    vec0_ = torch.normal(0, 1, size=(3,), dtype=torch.float64)
    vec1_ = torch.normal(0, 1, size=(3,), dtype=torch.float64)
    vec2_ = torch.cross(vec0_, vec1_, dim=-1)
    vec3_ = torch.cross(vec1_, vec2_, dim=-1)

    vec1 = vec1_/torch.linalg.vector_norm(vec1_)
    vec2 = vec2_/torch.linalg.vector_norm(vec2_)
    vec3 = vec3_/torch.linalg.vector_norm(vec3_)

    ret = torch.stack([vec1, vec2, vec3]).to(torch.float32)
    return ret

def rotate(matrix, scalar):
    scalar = scalar.transpose(-1,-2)
    scalar = scalar @ matrix
    scalar = scalar.transpose(-1,-2)
    return scalar

def test_inveqv_v(testfun, vector):
    delta_v = testfun(vector)
    for i in range(N_TEST):
        A = gen_rotate_matrix()
        v_rot = rotate(A, vector)
        delta_v_new = testfun(v_rot)
        delta_v_new_rot = rotate(A.T, delta_v_new)

        torch.testing.assert_close(delta_v, delta_v_new_rot)  # Equivariancy and invariancy check
        assert (torch.all(torch.isfinite(delta_v_new_rot)))

        assert (torch.sum(torch.abs(delta_v_new_rot)) > NON_DEGENERACY_TOLERANCE)

def test_inv_v(testfun, vector):
    delta_v = testfun(vector)
    for i in range(N_TEST):
        A = gen_rotate_matrix()
        v_rot = rotate(A, vector)
        delta_v_new = testfun(v_rot)

        torch.testing.assert_close(delta_v, delta_v_new)  # Equivariancy and invariancy check
        assert (torch.all(torch.isfinite(delta_v_new)))

        assert (torch.sum(torch.abs(delta_v_new)) > NON_DEGENERACY_TOLERANCE)

def test_inveqv_sv(testfun, scalar, vector):
    delta_s, delta_v = testfun(scalar, vector)
    for i in range(N_TEST):
        A = gen_rotate_matrix()
        v_rot = rotate(A, vector)
        delta_s_new, delta_v_new = testfun(scalar, v_rot)
        delta_v_new_rot = rotate(A.T, delta_v_new)

        torch.testing.assert_close(delta_s, delta_s_new)  # Invariancy check
        torch.testing.assert_close(delta_v, delta_v_new_rot)  # Equivariancy check
        assert (torch.all(torch.isfinite(delta_s_new)))
        assert (torch.all(torch.isfinite(delta_v_new_rot)))

        assert (torch.sum(torch.abs(delta_s_new))> NON_DEGENERACY_TOLERANCE)
        assert (torch.sum(torch.abs(delta_v_new_rot)) > NON_DEGENERACY_TOLERANCE)

def test_2outputs_inveqv_sv(testfun, scalar, vector, scalar2, vector2):
    delta_s, delta_v = testfun(scalar, vector, scalar2, vector2)
    for i in range(N_TEST):
        A = gen_rotate_matrix()
        v_rot = rotate(A, vector)
        v2_rot = rotate(A, vector2)
        delta_s_new, delta_v_new = testfun(scalar, v_rot, scalar2, v2_rot)
        delta_v_new_rot = rotate(A.T, delta_v_new)

        torch.testing.assert_close(delta_s, delta_s_new)  # Invariancy check
        torch.testing.assert_close(delta_v, delta_v_new_rot)  # Equivariancy check
        assert (torch.all(torch.isfinite(delta_s_new)))
        assert (torch.all(torch.isfinite(delta_v_new_rot)))

        assert (torch.sum(torch.abs(delta_s_new))> NON_DEGENERACY_TOLERANCE)
        assert (torch.sum(torch.abs(delta_v_new_rot)) > NON_DEGENERACY_TOLERANCE)

def test_2outputs_inveqv_vv(testfun, vector, vector2):
    delta_s, delta_v = testfun(vector, vector2)
    for i in range(N_TEST):
        A = gen_rotate_matrix()
        v_rot = rotate(A, vector)
        v2_rot = rotate(A, vector2)
        delta_s_new, delta_v_new = testfun(v_rot, v2_rot)
        delta_v_new_rot = rotate(A.T, delta_v_new)

        torch.testing.assert_close(delta_s, delta_s_new)  # Invariancy check
        torch.testing.assert_close(delta_v, delta_v_new_rot)  # Equivariancy check
        assert (torch.all(torch.isfinite(delta_s_new)))
        assert (torch.all(torch.isfinite(delta_v_new_rot)))

        assert (torch.sum(torch.abs(delta_s_new))> NON_DEGENERACY_TOLERANCE)
        assert (torch.sum(torch.abs(delta_v_new_rot)) >NON_DEGENERACY_TOLERANCE)

def test_shape_and_nondegeneracy(expected_shape, tensor):
    assert (expected_shape == tuple(tensor.shape))
    assert (torch.all(torch.isfinite(tensor)))
    assert (torch.sum(torch.abs(tensor)) > NON_DEGENERACY_TOLERANCE)
