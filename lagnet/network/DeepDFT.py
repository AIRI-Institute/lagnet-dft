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

from typing import Any

import torch
import pytorch_lightning as L
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..metric import NMAE_avg, NMAE_max, UniversalMetric
from ..layer import DeepDFText0_Model, DeepDFTinv_Model

from ..layer.DeepDFTeqv import DeepDFTeqv_Model
from ..third_party import ShiftedSoftplus


class DeepDFTinv_Wrapper(torch.nn.Module):

    def __init__(self, name='DeepDFTinv',
                       cutoff=15,
                       num_features=32,
                       size_of_expansion=20,
                       depth=2,
                       atom_embedding_size=36):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self.deep_dft = DeepDFTinv_Model(cutoff=cutoff,
                                         depth=depth,
                                         size_of_expansion=size_of_expansion,
                                         num_features=num_features,
                                         atom_embedding_size=atom_embedding_size)

        self.mlp_probing = torch.nn.Sequential(torch.nn.Linear(num_features, num_features),
                                                ShiftedSoftplus(),
                                                torch.nn.Linear(num_features, 1))

    def forward(self, batch):
        atom_num = batch['atom_num'].to(torch.int64)
        atom_coords = batch['atom_coords'].to(torch.float32)
        probe_cooords = batch['probe_cooords'].to(torch.float32)

        scalar = self.deep_dft(atom_coords, probe_cooords, atom_num)
        probe_values = self.mlp_probing(scalar.squeeze()).squeeze()

        return probe_values


class DeepDFTeqv_Wrapper(torch.nn.Module):

    def __init__(self, name='DeepDFTeqv',
                       cutoff=15,
                       num_features=32,
                       size_of_expansion=20,
                       atom_embedding_size=36,
                       depth=2):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self.deep_dft = DeepDFTeqv_Model(cutoff=cutoff,
                                         depth=depth,
                                         size_of_expansion=size_of_expansion,
                                         num_features=num_features,
                                         atom_embedding_size=atom_embedding_size)

        self.mlp_probing = torch.nn.Sequential(torch.nn.Linear(self.num_features, self.num_features),
                                               torch.nn.SiLU(),
                                               torch.nn.Linear(self.num_features, 1))

    def forward(self, batch):
        atom_num = batch['atom_num'].to(torch.int64)
        atom_coords = batch['atom_coords'].to(torch.float32)
        probe_cooords = batch['probe_cooords'].to(torch.float32)

        scalar, vector = self.deep_dft(atom_coords, probe_cooords, atom_num)
        probe_values = self.mlp_probing(scalar.squeeze()).squeeze()

        return probe_values


class DeepDFText0_Wrapper(torch.nn.Module):

    def __init__(self, name='DeepDFText0',
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
                         deepdft_column1_basis='fourier'):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self.deep_dft = DeepDFText0_Model(depth=depth,
                                         size_of_expansion=size_of_expansion,
                                         num_features=num_features,
                                         atom_embedding_size=atom_embedding_size,
                                         painn_column_cutoff=painn_column_cutoff,
                                         painn_column_min_value=painn_column_min_value,
                                         painn_column_max_value=painn_column_max_value,
                                         painn_column_basis=painn_column_basis,
                                         deepdft_column0_cutoff=deepdft_column0_cutoff,
                                         deepdft_column0_min_value=deepdft_column0_min_value,
                                         deepdft_column0_max_value=deepdft_column0_max_value,
                                         deepdft_column0_basis=deepdft_column0_basis,
                                         deepdft_column1_cutoff=deepdft_column1_cutoff,
                                         deepdft_column1_min_value=deepdft_column1_min_value,
                                         deepdft_column1_max_value=deepdft_column1_max_value,
                                         deepdft_column1_basis=deepdft_column1_basis)

        self.mlp_probing = torch.nn.Sequential(torch.nn.Linear(4 * self.num_features, 2 * self.num_features),
                                               torch.nn.SiLU(),
                                               torch.nn.Linear(2 * self.num_features, self.num_features),
                                               torch.nn.SiLU(),
                                               torch.nn.Linear(self.num_features, 1))

    def forward(self, batch):
        atom_num = batch['atom_num'].to(torch.int64)
        atom_coords = batch['atom_coords'].to(torch.float32)
        probe_cooords = batch['probe_cooords'].to(torch.float32)

        scalar, vector, p1_scalar, p1_vector, p2_scalar, p2_vector = self.deep_dft(atom_coords, probe_cooords, atom_num)
        probe_features = torch.concatenate([p1_scalar,
                                                    torch.linalg.vector_norm(p1_vector, axis=-2, keepdim=True),
                                                    p2_scalar,
                                                    torch.linalg.vector_norm(p2_vector, axis=-2, keepdim=True)], axis=-1)
        probe_values = self.mlp_probing(probe_features).squeeze()

        return probe_values


class DeepDFT_LModel(L.LightningModule):
    def __init__(self, cfg, model, test_only=False, path_to_save_metric = None):
        super().__init__()
        self.cfg = cfg

        self.test_only = test_only
        if test_only:
            self.automatic_optimization = False
            self.device_ = torch.device(cfg.test.device)
        else:
            self.automatic_optimization = True
            self.device_ = torch.device(cfg.train.device)

        self.model = model.to(self.device_)

        self.loss = torch.nn.MSELoss()
        #self.WATER = test_water_(cfg.file.atomization_energy, self.device_).WATER()

        self.train_MSE = torchmetrics.regression.MeanSquaredError()
        self.val_MSE = torchmetrics.regression.MeanSquaredError()

        self.train_MAE = torchmetrics.regression.MeanAbsoluteError()
        self.val_MAE = torchmetrics.regression.MeanAbsoluteError()

        self.train_MAPE = torchmetrics.regression.MeanAbsolutePercentageError()
        self.val_MAPE = torchmetrics.regression.MeanAbsolutePercentageError()

        self.val_MAE_el = torchmetrics.regression.MeanAbsoluteError()
        self.val_NMAE_avg = NMAE_avg(batched_data=False)
        self.val_NMAE2_avg = NMAE_avg(batched_data=False)

        self.val_NMAE_max = NMAE_max(batched_data=False)
        self.val_NMAE2_max = NMAE_max(batched_data=False)

        self.path_to_save_metric = path_to_save_metric
        self.universal_metric = UniversalMetric()
        self.universal_metric_core_orbitals = UniversalMetric(name_prefix='core_')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         patience=self.cfg.train.ReduceLROnPlateau_patience,
                                         factor=self.cfg.train.ReduceLROnPlateau_factor)
        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": True,
            "monitor": self.cfg.train.ReduceLROnPlateau_monitor,
            "patience": self.cfg.train.ReduceLROnPlateau_patience,
            "mode": "min",
            "factor": self.cfg.train.ReduceLROnPlateau_factor,
            "verbose": True,
            "interval": "epoch",
            "frequency": self.cfg.train.validation_frequency
        }

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        pred_el_in_point = self.model(train_batch)
        exp_el_in_point = train_batch['probe_target'].squeeze()
        loss = torch.mean((pred_el_in_point - exp_el_in_point) ** 2)

        self.log('loss', loss, on_step=True, prog_bar=True)
        bv_MSE = self.train_MSE(pred_el_in_point.flatten(), exp_el_in_point.flatten())
        bv_MAE = self.train_MAE(pred_el_in_point.flatten(), exp_el_in_point.flatten())
        bv_MAPE = self.train_MAPE(pred_el_in_point.flatten(), exp_el_in_point.flatten())

        self.log('train_MSE', bv_MSE, on_step=True, prog_bar=True)
        self.log('train_MAE', bv_MAE, on_step=True, prog_bar=True)
        self.log('train_MAPE', bv_MAPE, on_step=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.train_MSE.reset()
        self.train_MAE.reset()
        self.train_MAPE.reset()

    def _additinal_test_step(self, val_batch, batch_idx, pred_el_in_point):
        core_multiplier = val_batch['core_multiplier'].squeeze()
        probe_weights = val_batch['probe_weights'].squeeze()
        probe_target_original = val_batch['probe_target_original'].squeeze()

        unique_id = val_batch['unique_id'].squeeze()
        atom_num = val_batch['atom_num'].squeeze()

        self.universal_metric_core_orbitals.update(core_multiplier*pred_el_in_point,
                                                   probe_target_original,
                                                   probe_weights,
                                                   atom_num,
                                                   unique_id,
                                                   batch_idx)

    def _shared_validationa_and_test_step(self, val_batch, batch_idx):
        pred_el_in_point = self.model(val_batch)
        exp_el_in_point = val_batch['probe_target'].squeeze()
        probe_weights = val_batch['probe_weights'].squeeze()
        loss = torch.mean((pred_el_in_point - exp_el_in_point) ** 2)

        self.log('loss', loss, on_step=True, prog_bar=True)
        self.val_MSE.update(pred_el_in_point.flatten(), exp_el_in_point.flatten())
        self.val_MAE.update(pred_el_in_point.flatten(), exp_el_in_point.flatten())
        self.val_MAPE.update(pred_el_in_point.flatten(), exp_el_in_point.flatten())

        self.val_MAE_el.update(probe_weights.flatten() * pred_el_in_point.flatten(),
                               probe_weights.flatten() * exp_el_in_point.flatten())
        self.val_NMAE_avg.update(pred_el_in_point, exp_el_in_point, probe_weights)
        self.val_NMAE2_avg.update(pred_el_in_point, exp_el_in_point)
        self.val_NMAE_max.update(pred_el_in_point, exp_el_in_point, probe_weights)
        self.val_NMAE2_max.update(pred_el_in_point, exp_el_in_point)

        unique_id = val_batch['unique_id'].squeeze()
        atom_num = val_batch['atom_num'].squeeze()
        self.universal_metric.update(pred_el_in_point, exp_el_in_point, probe_weights, atom_num, unique_id,
                                                   batch_idx)

        if self.test_only:
            self._additinal_test_step(val_batch, batch_idx, pred_el_in_point)

    def validation_step(self, val_batch, batch_idx):
        self._shared_validationa_and_test_step(val_batch, batch_idx)

    def test_step(self, val_batch, batch_idx):
        torch.set_grad_enabled(False)
        self._shared_validationa_and_test_step(val_batch, batch_idx)
        torch.set_grad_enabled(True)


    def _shared_validation_and_test_epoch_end(self):

        self.log('val_MSE', self.val_MSE.compute(), prog_bar=True)
        self.log('val_MAE', self.val_MAE.compute(), prog_bar=True)
        self.log('val_MAPE', self.val_MAPE.compute(), prog_bar=True)
        self.log('val_MAE_el', self.val_MAE_el.compute(), prog_bar=True)
        NMAE_avg_, NMAE_std_ = self.val_NMAE_avg.compute()
        self.log('val_NMAE_avg', NMAE_avg_, prog_bar=True)
        self.log('val_NMAE_std', NMAE_std_, prog_bar=False)
        NMAE2_avg_, NMAE2_std_ = self.val_NMAE2_avg.compute()
        self.log('val_NMAE2_avg', NMAE2_avg_, prog_bar=True)
        self.log('val_NMAE2_std', NMAE2_std_, prog_bar=False)
        self.log('val_NMAE_max', self.val_NMAE_max.compute(), prog_bar=False)
        self.log('val_NMAE2_max', self.val_NMAE2_max.compute(), prog_bar=False)
        if not self.test_only:
            self.log('lr', self.lr_schedulers().optimizer.param_groups[0]['lr'])


        self.val_MSE.reset()
        self.val_MAE.reset()
        self.val_MAPE.reset()
        self.val_MAE_el.reset()

        self.val_NMAE_avg.reset()
        self.val_NMAE2_avg.reset()
        self.val_NMAE_max.reset()
        self.val_NMAE2_max.reset()

    def on_validation_epoch_end(self):
        self._shared_validation_and_test_epoch_end()

        if self.cfg.train.max_epochs > 0:
            if self.current_epoch > self.cfg.train.max_epochs:
                self.trainer.should_stop = True

        if (not self.test_only) and self.lr_schedulers().optimizer.param_groups[0]['lr'] < self.cfg.train.lr_stop:
            self.trainer.should_stop = True

    def on_test_epoch_end(self):
        self._shared_validation_and_test_epoch_end()
        if self.path_to_save_metric is not None:
            df, df_full = self.universal_metric.get_dataframe()
            df2, df2_full = self.universal_metric_core_orbitals.get_dataframe()
            df_final = df.join(df2)
            df_final_full = df_full.join(df2_full)
            df_final.to_csv(self.path_to_save_metric+"_RAW.csv")
            df_final_full.to_csv(self.path_to_save_metric)

        self.universal_metric.reset()
        self.universal_metric_core_orbitals.reset()
