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

from typing import Tuple

import torch
from torchmetrics import Metric
import pandas as pd

class UniversalMetric(Metric):
    def __init__(self, epsilon: float = 10e-21, name_prefix='', **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.name_prefix = name_prefix

        self.add_state("electron_target", default=[], dist_reduce_fx="cat")
        self.add_state("electron_predicted", default=[], dist_reduce_fx="cat")

        self.add_state("dividend_list", default=[], dist_reduce_fx="cat")
        self.add_state("divisor_list", default=[], dist_reduce_fx="cat")
        self.add_state("weighted_dividend_list", default=[], dist_reduce_fx="cat")
        self.add_state("weighted_divisor_list", default=[], dist_reduce_fx="cat")

        self.add_state("negative_values", default=[], dist_reduce_fx="cat")
        self.add_state("weighted_negative_values", default=[], dist_reduce_fx="cat")

        self.add_state("exact_number_of_electron", default=[], dist_reduce_fx="cat")
        self.add_state("unique_id", default=[], dist_reduce_fx="cat")
        self.add_state("point_number", default=[], dist_reduce_fx="cat")
        self.add_state("batch_index", default=[], dist_reduce_fx="cat")

    def update(self,
               predicted: torch.Tensor,
               target: torch.Tensor,
               weights: torch.Tensor,
               atom_numbers: torch.Tensor,
               mol_id:  torch.Tensor,
               batch_index:  torch.Tensor) -> None:
        if weights is None:
            weights = torch.ones_like(predicted)

        if predicted.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        if weights.shape != target.shape:
            raise ValueError("preds and weights must have the same shape")

        weighted_dividend = weights * torch.abs(predicted - target)
        weighted_divisor = weights * torch.abs(target)# + self.epsilon
        dividend = torch.abs(predicted - target)
        divisor = torch.abs(target)# + self.epsilon

        predicted_negative = torch.clip(predicted, None, 0.)
        weighted_predicted_negative = weights*torch.clip(predicted, None, 0.)

        self.electron_target.append((weights * target).sum())
        self.electron_predicted.append((weights * predicted).sum())

        self.weighted_dividend_list.append(weighted_dividend.sum())
        self.weighted_divisor_list.append(weighted_divisor.sum())
        self.dividend_list.append(dividend.sum())
        self.divisor_list.append(divisor.sum())

        self.negative_values.append(predicted_negative.sum())
        self.weighted_negative_values.append(weighted_predicted_negative.sum())

        self.exact_number_of_electron.append(atom_numbers.sum())
        self.unique_id.append(mol_id)
        self.point_number.append(len(target))
        self.batch_index.append(batch_index)


    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        NMAE = torch.tensor(self.weighted_dividend_list) / torch.tensor(self.weighted_divisor_list)
        NMAE2 = torch.tensor(self.dividend_list) / torch.tensor(self.divisor_list)

        negative_values = torch.tensor(self.negative_values)
        weighted_negative_values = torch.tensor(self.weighted_negative_values)

        electron_target = torch.tensor(self.electron_target)
        electron_predicted = torch.tensor(self.electron_predicted)

        dict_ = {self.name_prefix + "NMAE_avg": torch.mean(NMAE),
                 self.name_prefix + "NMAE_std": torch.std(NMAE),
                 self.name_prefix + "NMAE2_avg": torch.mean(NMAE2),
                 self.name_prefix + "NMAE2_std": torch.std(NMAE2),
                 self.name_prefix + "electron_target_avg": torch.mean(electron_target),
                 self.name_prefix + "electron_target_std": torch.std(electron_target),
                 self.name_prefix + "electron_predicted_avg": torch.mean(electron_predicted),
                 self.name_prefix + "electron_predicted_std": torch.std(electron_predicted),
                 self.name_prefix + "negative_values_avg": torch.mean(negative_values),
                 self.name_prefix + "negative_values_std": torch.std(negative_values),
                 self.name_prefix + "weighted_negative_values_avg": torch.mean(weighted_negative_values),
                 self.name_prefix + "weighted_negative_values_std": torch.std(weighted_negative_values)}

        return dict_

    def get_dataframe(self):
        weighted_dividend_list = torch.tensor(self.weighted_dividend_list)
        weighted_divisor_list = torch.tensor(self.weighted_divisor_list)
        dividend_list = torch.tensor(self.dividend_list)
        divisor_list = torch.tensor(self.divisor_list)

        negative_values = torch.tensor(self.negative_values)
        weighted_negative_values = torch.tensor(self.weighted_negative_values)

        exact_number_of_electron = torch.tensor(self.exact_number_of_electron)
        unique_id = torch.tensor(self.unique_id)

        electron_target = torch.tensor(self.electron_target)
        electron_predicted = torch.tensor(self.electron_predicted)
        point_number = torch.tensor(self.point_number)
        batch_index = torch.tensor(self.batch_index)

        NMAE = weighted_dividend_list / (weighted_divisor_list + self.epsilon)
        NMAE2 = dividend_list / (divisor_list + self.epsilon)

        df = pd.DataFrame({self.name_prefix+"NMAE": NMAE.tolist(),
                           self.name_prefix+"NMAE2": NMAE2.tolist(),
                           self.name_prefix+"weighted_dividend": weighted_dividend_list.tolist(),
                           self.name_prefix+"weighted_divisor": weighted_divisor_list.tolist(),
                           self.name_prefix+"dividend": dividend_list.tolist(),
                           self.name_prefix+"divisor": divisor_list.tolist(),
                           self.name_prefix+"electron_target": electron_target.tolist(),
                           self.name_prefix+"electron_predicted": electron_predicted.tolist(),
                           self.name_prefix+"negative_values": negative_values.tolist(),
                           self.name_prefix+"weighted_negative_values": weighted_negative_values.tolist(),
                           self.name_prefix+"exact_number_of_electron": exact_number_of_electron.tolist(),
                           self.name_prefix+"unique_id": unique_id.tolist(),
                           self.name_prefix+"point_number": point_number.tolist(),
                           self.name_prefix+"batch_index": batch_index.tolist()})

        list_ = []
        for unique_id in df[self.name_prefix+"unique_id"].unique():
            subset = df[df[self.name_prefix + "unique_id"] == unique_id]
            weighted_dividend_list = subset[self.name_prefix + "weighted_dividend"].sum()
            weighted_divisor = subset[self.name_prefix + "weighted_divisor"].sum()

            dividend_list = subset[self.name_prefix + "dividend"].sum()
            divisor_list = subset[self.name_prefix + "divisor"].sum()

            electron_target = subset[self.name_prefix + "electron_target"].sum()
            electron_predicted = subset[self.name_prefix + "electron_predicted"].sum()
            negative_values = subset[self.name_prefix + "negative_values"].sum()
            weighted_negative_values = subset[self.name_prefix + "weighted_negative_values"].sum()
            exact_number_of_electron = subset[self.name_prefix + "exact_number_of_electron"].iloc[0]

            point_number = subset[self.name_prefix + "point_number"].sum()

            dictionary = {self.name_prefix + "NMAE": weighted_dividend_list/weighted_divisor,
                           self.name_prefix + "NMAE2": dividend_list/divisor_list,
                           self.name_prefix + "weighted_dividend": weighted_dividend_list,
                           self.name_prefix + "weighted_divisor": weighted_divisor,
                           self.name_prefix + "dividend": dividend_list,
                           self.name_prefix + "divisor": divisor_list,
                           self.name_prefix + "electron_target": electron_target,
                           self.name_prefix + "electron_predicted": electron_predicted,
                           self.name_prefix + "negative_values": negative_values,
                           self.name_prefix + "weighted_negative_values": weighted_negative_values,
                           self.name_prefix + "exact_number_of_electron": exact_number_of_electron,
                           self.name_prefix + "unique_id": unique_id,
                           self.name_prefix + "point_number": point_number,
                           self.name_prefix + "batch_index": 0}
            list_.append(dictionary)

        df2 = pd.DataFrame(list_)

        return df, df2


class ConcatNMAE_avg(Metric):
    def __init__(self, epsilon: float = 10e-21, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

        self.add_state("dividend_list", default=[], dist_reduce_fx="cat")
        self.add_state("divisor_list", default=[], dist_reduce_fx="cat")
        self.add_state("unique_id", default=[], dist_reduce_fx="cat")

    def update(self, unique_id : torch.int64, predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> None:
        if predicted.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        if weights is not None and weights.shape != target.shape:
            raise ValueError("preds and weights must have the same shape")

        if weights is not None:
            dividend = weights * torch.abs(predicted - target)
            divisor = weights * torch.abs(target) + self.epsilon
        else:
            dividend = torch.abs(predicted - target)
            divisor = torch.abs(target) + self.epsilon

        self.dividend_list.append(dividend.sum())
        self.divisor_list.append(divisor.sum())
        self.unique_id.append(unique_id)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:

        unique_index = list(set(self.unique_id))

        sum_x1 = torch.tensor(0., requires_grad=False)
        sum_x2 = torch.tensor(0., requires_grad=False)
        total = torch.tensor(0., requires_grad=False)

        for ind in unique_index:
            dividend_ = torch.tensor(0., requires_grad=False)
            divisor_ = torch.tensor(0., requires_grad=False)
            for dividend, divisor, unique_id in zip(self.dividend_list, self.divisor_list, self.unique_id):
                if unique_id.item() == ind.item():
                    dividend_ += dividend
                    divisor_ += divisor

            sum_x1 += (dividend_/divisor_)
            sum_x2 += (dividend_/divisor_)**2
            total += 1

        return sum_x1 / total, torch.sqrt((sum_x2 / total) - (sum_x1 / total)**2)


class NMAE_avg(Metric):
    def __init__(self, batched_data: bool = True, epsilon: float = 10e-21, **kwargs):
        super().__init__(**kwargs)
        self.batched_data = batched_data
        self.epsilon = epsilon

        self.add_state("sum_x1", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> None:
        if predicted.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        if weights is not None and weights.shape != target.shape:
            raise ValueError("preds and weights must have the same shape")

        if weights is not None:
            dividend = weights * torch.abs(predicted - target)
            divisor = weights * torch.abs(target) + self.epsilon
        else:
            dividend = torch.abs(predicted - target)
            divisor = torch.abs(target) + self.epsilon

        if self.batched_data:
            dividend = dividend.sum(tuple(range(1, dividend.ndim)))
            divisor = divisor.sum(tuple(range(1, divisor.ndim)))
            self.sum_x1 += torch.sum(dividend / divisor)
            self.sum_x2 += torch.sum((dividend / divisor)**2)
            self.total += target.shape[0]
        else:
            dividend = dividend.sum()
            divisor = divisor.sum()
            self.sum_x1 += (dividend / divisor)
            self.sum_x2 += (dividend / divisor)**2
            self.total += 1

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sum_x1 / self.total, torch.sqrt((self.sum_x2 / self.total) - (self.sum_x1 / self.total)**2)



class NMAE_max(Metric):
    def __init__(self, batched_data: bool = True, epsilon: float = 10e-21, **kwargs):
        super().__init__(**kwargs)
        self.batched_data = batched_data
        self.epsilon = epsilon

        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="max")

    def update(self, predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> None:
        if predicted.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        if weights is not None and weights.shape != target.shape:
            raise ValueError("preds and weights must have the same shape")

        if weights is not None:
            dividend = weights * torch.abs(predicted - target)
            divisor = weights * torch.abs(target) + self.epsilon
        else:
            dividend = torch.abs(predicted - target)
            divisor = torch.abs(target) + self.epsilon

        if self.batched_data:
            dividend = dividend.sum(tuple(range(1, dividend.ndim)))
            divisor = divisor.sum(tuple(range(1, divisor.ndim)))
            self.correct = max(self.correct, torch.max(dividend / divisor))
        else:
            dividend = dividend.sum()
            divisor = divisor.sum()
            self.correct = max(self.correct, (dividend / divisor))

    def compute(self) -> torch.Tensor:
        return self.correct