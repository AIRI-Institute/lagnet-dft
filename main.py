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

import os
import random
import torch

import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as L


from omegaconf import OmegaConf
import hydra
import lagnet
from lagnet.dataloader import Nabla2DFTDataset, Nabla2DFTDatasetSGPerMolecule, Nabla2DFTDatasetUniformGrid, DatasetRegistry, \
    ModelRegistry, Nabla2DFTDatasetPointCloud
from lagnet.network.DeepDFT import DeepDFT_LModel

import hruid
from datetime import datetime

from lagnet.third_party import HamiltonianDatabase

def test2(cfg):
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output_folder, exist_ok=True)

    data_storage = DatasetRegistry(cfg.data_path)
    checkpoint_storage = ModelRegistry(cfg.checkpoint_path)

    print("List of available checkpoints:")
    print(checkpoint_storage.get_checkpoint_list())
    print("List of available datasets:")
    print(data_storage.get_data_list())

    tmp_ckp = checkpoint_storage.get_checkpoint(cfg.checkpoint_key)
    checkpoint_file_path = os.path.join(cfg.checkpoint_path, tmp_ckp['file'])
    checkpoint_file = tmp_ckp['file']
    tmp_dt = data_storage.get_data(cfg.data_key)
    target_files = os.path.join(cfg.data_path, tmp_dt['target_files'])
    hamiltonian_db = os.path.join(cfg.data_path, tmp_dt['database'])
    list_of_files = os.path.join(cfg.data_path, tmp_dt['list_of_files'])
    data_format = tmp_dt['data_format']

    list_of_index = np.load(list_of_files)[:cfg.test.dataset_subset_size]

    if data_format == 'sg_per_atom':
        dts2 = DataLoader(Nabla2DFTDataset(list_of_index,
                                           cfg.atomization_energy,
                                           target_files,
                                           hamiltonian_db,
                                           sampling_rate=None,
                                           is_test=True,
                                           is_supress_cores=cfg.is_supress_cores),
                          batch_size=1,
                          shuffle=False,
                          num_workers=cfg.num_workers_test)
    elif data_format == 'sg_per_molecule':
        dts2 = DataLoader(Nabla2DFTDatasetSGPerMolecule(list_of_index,
                                                        cfg.atomization_energy,
                                                        target_files,
                                                        hamiltonian_db,
                                                        is_supress_cores=cfg.is_supress_cores),
                          batch_size=1,
                          shuffle=False,
                          num_workers=cfg.num_workers_test)
    elif data_format == 'uniform_grid':
        dts2 = DataLoader(Nabla2DFTDatasetUniformGrid(list_of_index,
                                                      cfg.atomization_energy,
                                                      target_files,
                                                      hamiltonian_db,
                                                      is_supress_cores=cfg.is_supress_cores),
                          batch_size=1,
                          shuffle=False,
                          num_workers=cfg.num_workers_test)

    elif data_format == 'point_cloud':
        dts2 = DataLoader(Nabla2DFTDatasetPointCloud(list_of_index,
                                                      cfg.atomization_energy,
                                                      target_files,
                                                      hamiltonian_db,
                                                      is_supress_cores=cfg.is_supress_cores),
                          batch_size=1,
                          shuffle=False,
                          num_workers=cfg.num_workers_test)
    else:
        raise NotImplementedError('Format for data preparation are unknown')

    model: torch.nn.Module = hydra.utils.instantiate(cfg.nn)
    device_ = torch.device(cfg.test.device)
    checkpoint = torch.load(checkpoint_file_path, map_location=lambda storage, loc: storage)

    # Spikes
    prefix = 'model.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint["state_dict"].items()
                    if k.startswith(prefix)}

    model.load_state_dict(adapted_dict)
    model = model.to(device_)
    lightning_module = DeepDFT_LModel(cfg,
                                      model,
                                      test_only=True,
                                      path_to_save_metric = os.path.join(cfg.output_folder,
                                                                         f'{cfg.checkpoint_key}-{cfg.data_key}.csv'))

    trainer = L.Trainer(accelerator=cfg.test.accelerator)
    tmp_ = trainer.test(lightning_module, dataloaders=dts2)
    dictionary_to_store = dict(tmp_[0])
    dictionary_to_store['nn_name'] = cfg.checkpoint_key
    dictionary_to_store['data_name'] = cfg.data_key
    dictionary_to_store['test_name'] = cfg.test.test_name
    dictionary_to_store['description'] = cfg.test.description
    dictionary_to_store['checkpoint'] = checkpoint_file
    dictionary_to_store['code_version'] = lagnet.__version__
    dictionary_to_store['NMAE_array'] = f'{cfg.checkpoint_key}-{cfg.data_key}.csv'

    with open(os.path.join(cfg.output_folder, f"{cfg.checkpoint_key}-{cfg.data_key}.json"), 'w') as fl:
        json.dump(dictionary_to_store, fl)

@hydra.main(config_path="./config", config_name='article_test2', version_base="1.2")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    if cfg.job_type == 'train':
        NotImplementedError("Will be released soon")
    elif cfg.job_type == 'test':
        NotImplementedError("Will be released soon")
    elif cfg.job_type == 'test2':
        test2(cfg)
    elif cfg.job_type == 'prepare_dataset':
        NotImplementedError("Will be released soon")
    else:
        raise NotImplementedError(f"Job type `{cfg.job_type}` is not implemented")

if __name__ == "__main__":
    my_app()