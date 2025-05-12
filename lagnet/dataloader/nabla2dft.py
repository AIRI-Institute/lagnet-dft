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

import numpy as np
import torch
from torch.utils.data import Dataset
import os

from ase.data import covalent_radii
from ase.units import Bohr

from sklearn.metrics.pairwise import pairwise_distances

from ..third_party import HamiltonianDatabase
from tqdm import tqdm

Nabla2DFT_ATOMS_NUM = (1,6,7,8,9,16,17,35)

def _sigmoid(z):
    return 1/(1 + np.exp(-z))

class CoreSupressionMethod:
    DICT_OF_PARAMS = {6: (10.427195105452354,
                          -0.30207123849405104,
                          0.752070155497629,
                          3.2902825700737384,
                          5.2572758345059025),
                      7: (7.082862195412066,
                          -0.1954490301084247,
                          -0.48668064582656995,
                          2.664846696860429,
                          3.823191993159255),
                      8: (17.952233173315587,
                          -0.2523860221832123,
                          -0.6416376517077778,
                          4.002781975515073,
                          5.974346267452812),
                      9: (26.034518760253494,
                          -0.2529622382294457,
                          -0.6898156659102556,
                          4.391208824783542,
                          7.7851465597178775),
                      16: (89.37072995798225,
                           -0.191078052526283,
                           -0.7124757193195426,
                           6.161838576838525,
                           -11.186638263119109),
                      17: (88.64169607191243,
                           -0.1733533419258262,
                           0.6470192116925113,
                           6.034675361800486,
                           10.562997262676518),
                      35: (161.0282978186252,
                           -0.07546882331958502,
                           -0.5249849213355682,
                           4.6006322421017956,
                           18.697134292575978)}

    def __init__(self):
        pass

    def join_function(self, supress_multiplier, distance, values, FIT_DIST):
        div = np.maximum(supress_multiplier * _sigmoid(1 * (FIT_DIST - distance)) + 1., 1.)
        return values / div, div

    def process(self, Z, distance, probe_values):

        A = self.DICT_OF_PARAMS[Z][0]
        w0 = self.DICT_OF_PARAMS[Z][1]
        w1 = self.DICT_OF_PARAMS[Z][2]
        w2 = self.DICT_OF_PARAMS[Z][3]
        w3 = self.DICT_OF_PARAMS[Z][4]

        FIT_DIST = covalent_radii[Z] / Bohr

        supress_multiplier = A / (
                    (w0 ** 2) + (w1 ** 2) * distance + (w2 ** 2) * distance ** 2 + (w3 ** 2) * distance ** 3)


        ret, mul_local = self.join_function(supress_multiplier, distance, probe_values, FIT_DIST)
        return ret, mul_local


class Nabla2DFTDataset(Dataset):

    def __init__(self,
                 index_set,
                 atomization_energy_file,
                 path_target_val, path_db,
                 sampling_rate=None,
                 is_supress_cores=True,
                 is_test=False):
        self.path_target_val = path_target_val
        self.path_db = path_db

        self.index_set = index_set
        self.ham_db = HamiltonianDatabase(self.path_db)
        self.sampling_rate = sampling_rate

        assert len(self.ham_db) >= len(self.index_set)
        assert (sampling_rate is None) or (0 <= sampling_rate <= 1)

        self.core_supressor = CoreSupressionMethod()
        self.is_supress_cores = is_supress_cores
        self.atomization_energy = np.load(atomization_energy_file)

        self.is_test = is_test

    def __len__(self):
        return len(self.index_set)

    def __getitem__(self, index):
        Z, R, E, F, H, S, C, moses_id, conformation_id = self.ham_db[self.index_set[index]]
        atom_dict = np.load(os.path.join(self.path_target_val, f'{self.index_set[index]:06}.npy'),
                            allow_pickle=True).item()

        probe_cooords = []
        probe_weights = []
        probe_target = []

        if self.sampling_rate is None:
            for k, v in atom_dict.items():
                probe_cooords.append(v[0])
                probe_weights.append(v[1])
                probe_target.append(v[2])
        else:
            for k, v in atom_dict.items():
                pick_ind = np.random.choice(len(v[0]), int(np.round(self.sampling_rate * len(v[0]))), replace=False)
                probe_cooords.append(v[0][pick_ind])
                probe_weights.append(v[1][pick_ind])
                probe_target.append(v[2][pick_ind])

        probe_cooords = np.concatenate(probe_cooords)
        probe_weights = np.concatenate(probe_weights)
        probe_target = np.concatenate(probe_target)
        pw_distance = pairwise_distances(probe_cooords, R)
        unique_id = moses_id*1000000 + conformation_id
        core_multiplier = np.ones_like(probe_target)

        if self.is_test:
            probe_target_original = np.copy(probe_target)

        if self.is_supress_cores:
            for z, dist in zip(Z, pw_distance.T):
                if z > 1.1:  # Do not supress H
                    probe_target, loc_mul = self.core_supressor.process(z, dist, probe_target)
                    core_multiplier = core_multiplier*loc_mul

        ret_dict = {"atom_num": torch.tensor(Z),
                    "atom_coords": torch.tensor(R),
                    "probe_cooords": torch.tensor(probe_cooords),
                    "probe_weights": torch.tensor(probe_weights),
                    "probe_target": torch.tensor(probe_target),
                    "pw_distance": torch.tensor(pw_distance),
                    "unique_id": torch.tensor(unique_id, dtype=torch.int64)}

        if self.is_test:
            ret_dict["core_multiplier"] = torch.tensor(core_multiplier)
            ret_dict["probe_target_original"] = torch.tensor(probe_target_original)


        return ret_dict


class Nabla2DFTDatasetSGPerMolecule(Dataset):

    def __init__(self,
                 index_set,
                 atomization_energy_file,
                 path_target_val,
                 path_db,
                 is_supress_cores=True,
                 max_points=20000):
        self.path_target_val = path_target_val
        self.path_db = path_db
        self.max_points = max_points
        self.is_supress_cores=is_supress_cores
        self.core_supressor = CoreSupressionMethod()

        self.index_set = index_set
        self.ham_db = HamiltonianDatabase(self.path_db)
        self.atomization_energy = np.load(atomization_energy_file)

        self.data_locations = {}

        print("Preload test dataset")
        global_index = 0
        for index_ in tqdm(self.index_set):
            atom_dict = np.load(os.path.join(self.path_target_val,
                                             f'sg_per_molecule_{index_:06}.npz'),
                                allow_pickle=False)
            range_ = len(atom_dict['grid_weights'])

            for i in range(0, range_, max_points):
                self.data_locations[global_index] = {'file': f'sg_per_molecule_{index_:06}.npz',
                                                     'index': index_,
                                                     'start': i,
                                                     'finish': min(i+max_points, range_)}
                global_index += 1


    def __len__(self):
        return len(self.data_locations)


    def __getitem__(self, index):
        Z, R, E, F, H, S, C, moses_id, conformation_id = self.ham_db[self.data_locations[index]['index']]
        atom_dict = np.load(os.path.join(self.path_target_val, self.data_locations[index]['file']))

        start_ = self.data_locations[index]['start']
        finish_ = self.data_locations[index]['finish']
        probe_cooords = atom_dict['coords_bohr'][start_:finish_]
        probe_weights = atom_dict['grid_weights'][start_:finish_]
        probe_target = atom_dict['el_array'][start_:finish_]

        pw_distance = pairwise_distances(probe_cooords, R)
        unique_id = moses_id * 1000000 + conformation_id
        core_multiplier = np.ones_like(probe_target)
        probe_target_original = np.copy(probe_target)

        if self.is_supress_cores:
            for z, dist in zip(Z, pw_distance.T):
                if z > 1.1:  # Do not supress H
                    probe_target, loc_mul = self.core_supressor.process(z, dist, probe_target)
                    core_multiplier = core_multiplier * loc_mul

        ret_dict = {"atom_num": torch.tensor(Z),
                    "atom_coords": torch.tensor(R),
                    "probe_cooords": torch.tensor(probe_cooords),
                    "probe_weights": torch.tensor(probe_weights),
                    "probe_target": torch.tensor(probe_target),
                    "pw_distance": torch.tensor(pw_distance),
                    "unique_id": torch.tensor(unique_id, dtype=torch.int64)}

        ret_dict["core_multiplier"] = torch.tensor(core_multiplier)
        ret_dict["probe_target_original"] = torch.tensor(probe_target_original)

        return ret_dict


class Nabla2DFTDatasetPointCloud(Dataset):

    def __init__(self,
                 index_set,
                 atomization_energy_file,
                 path_target_val,
                 path_db,
                 is_supress_cores=True,
                 max_points=20000):
        self.path_target_val = path_target_val
        self.path_db = path_db
        self.max_points = max_points
        self.is_supress_cores=is_supress_cores
        self.core_supressor = CoreSupressionMethod()

        self.index_set = index_set
        self.ham_db = HamiltonianDatabase(self.path_db)
        self.atomization_energy = np.load(atomization_energy_file)

        self.data_locations = {}

        print("Preload test dataset")
        global_index = 0
        for index_ in tqdm(self.index_set):
            atom_dict = np.load(os.path.join(self.path_target_val,
                                             f'point_cloud_{index_:06}.npz'),
                                allow_pickle=False)
            range_ = len(atom_dict['grid_weights'])

            for i in range(0, range_, max_points):
                self.data_locations[global_index] = {'file': f'point_cloud_{index_:06}.npz',
                                                     'index': index_,
                                                     'start': i,
                                                     'finish': min(i+max_points, range_)}
                global_index += 1


    def __len__(self):
        return len(self.data_locations)


    def __getitem__(self, index):
        Z, R, E, F, H, S, C, moses_id, conformation_id = self.ham_db[self.data_locations[index]['index']]
        atom_dict = np.load(os.path.join(self.path_target_val, self.data_locations[index]['file']))

        start_ = self.data_locations[index]['start']
        finish_ = self.data_locations[index]['finish']
        probe_cooords = atom_dict['coords_bohr'][start_:finish_]
        probe_weights = atom_dict['grid_weights'][start_:finish_]
        probe_target = atom_dict['el_array'][start_:finish_]

        pw_distance = pairwise_distances(probe_cooords, R)
        unique_id = moses_id * 1000000 + conformation_id
        core_multiplier = np.ones_like(probe_target)
        probe_target_original = np.copy(probe_target)

        if self.is_supress_cores:
            for z, dist in zip(Z, pw_distance.T):
                if z > 1.1:  # Do not supress H
                    probe_target, loc_mul = self.core_supressor.process(z, dist, probe_target)
                    core_multiplier = core_multiplier * loc_mul

        ret_dict = {"atom_num": torch.tensor(Z),
                    "atom_coords": torch.tensor(R),
                    "probe_cooords": torch.tensor(probe_cooords),
                    "probe_weights": torch.tensor(probe_weights),
                    "probe_target": torch.tensor(probe_target),
                    "pw_distance": torch.tensor(pw_distance),
                    "unique_id": torch.tensor(unique_id, dtype=torch.int64)}

        ret_dict["core_multiplier"] = torch.tensor(core_multiplier)
        ret_dict["probe_target_original"] = torch.tensor(probe_target_original)

        return ret_dict


class Nabla2DFTDatasetUniformGrid(Dataset):

    def __init__(self,
                 index_set,
                 atomization_energy_file,
                 path_target_val,
                 path_db,
                 is_supress_cores=True,
                 max_points_per_side=30):
        self.path_target_val = path_target_val
        self.path_db = path_db
        self.max_points_per_side = max_points_per_side
        self.is_supress_cores=is_supress_cores
        self.core_supressor = CoreSupressionMethod()

        self.index_set = index_set
        self.ham_db = HamiltonianDatabase(self.path_db)
        self.atomization_energy = np.load(atomization_energy_file)

        self.data_locations = {}

        print("Preload test dataset")
        global_index = 0
        for index_ in tqdm(self.index_set):
            atom_dict = np.load(os.path.join(self.path_target_val,
                                             f'uniform_grid_{index_:06}.npz'),
                                allow_pickle=False)
            r_x, r_y, r_z = atom_dict['el_array'].shape

            for i in range(0, r_x, max_points_per_side):
                for j in range(0, r_y, max_points_per_side):
                    for k in range(0, r_z, max_points_per_side):
                        self.data_locations[global_index] = {'file': f'uniform_grid_{index_:06}.npz',
                                                             'index': index_,
                                                             'start_x': i, 'finish_x': min(i+max_points_per_side, r_x),
                                                             'start_y': j, 'finish_y': min(j+max_points_per_side, r_y),
                                                             'start_z': k, 'finish_z': min(k+max_points_per_side, r_z)}
                        global_index += 1


    def __len__(self):
        return len(self.data_locations)

    def __getitem__(self, index):
        Z, R, E, F, H, S, C, moses_id, conformation_id = self.ham_db[self.data_locations[index]['index']]
        atom_dict = np.load(os.path.join(self.path_target_val, self.data_locations[index]['file']))

        s_x, f_x = self.data_locations[index]['start_x'], self.data_locations[index]['finish_x']
        s_y, f_y = self.data_locations[index]['start_y'], self.data_locations[index]['finish_y']
        s_z, f_z = self.data_locations[index]['start_z'], self.data_locations[index]['finish_z']

        origin = atom_dict['origin']
        grid_volume = atom_dict['grid_volume']
        space_divisor = atom_dict['space_step']

        probe_target = atom_dict['el_array'][s_x:f_x,s_y:f_y,s_z:f_z].flatten()

        tmp_ = np.mgrid[s_x:f_x, s_y:f_y, s_z:f_z].reshape((3,-1)).T
        probe_cooords = space_divisor*tmp_ + origin
        probe_cooords = probe_cooords
        probe_weights = np.full(probe_cooords.shape[0], grid_volume)

        pw_distance = pairwise_distances(probe_cooords, R)
        unique_id = moses_id * 1000000 + conformation_id
        core_multiplier = np.ones_like(probe_target)
        probe_target_original = np.copy(probe_target)

        if self.is_supress_cores:
            for z, dist in zip(Z, pw_distance.T):
                if z > 1.1:  # Do not supress H
                    probe_target, loc_mul = self.core_supressor.process(z, dist, probe_target)
                    core_multiplier = core_multiplier * loc_mul

        ret_dict = {"atom_num": torch.tensor(Z),
                    "atom_coords": torch.tensor(R),
                    "probe_cooords": torch.tensor(probe_cooords),
                    "probe_weights": torch.tensor(probe_weights),
                    "probe_target": torch.tensor(probe_target),
                    "pw_distance": torch.tensor(pw_distance),
                    "unique_id": torch.tensor(unique_id, dtype=torch.int64)}

        ret_dict["core_multiplier"] = torch.tensor(core_multiplier)
        ret_dict["probe_target_original"] = torch.tensor(probe_target_original)

        return ret_dict





