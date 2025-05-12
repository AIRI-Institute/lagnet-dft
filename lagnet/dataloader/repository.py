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

import pandas as pd
import os

class ModelRegistry:

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Checkpoint path unavailable")
        self.path = path
        self.df = pd.read_csv(os.path.join(path, 'metadata.csv'))
        self.df['available'] = self.df.file.apply(lambda file: os.path.exists(os.path.join(path, file)))

    def get_checkpoint_list(self):
        return self.df[self.df.is_good & self.df.available].key.tolist()

    def get_checkpoint(self, name):
        if not name in self.get_checkpoint_list():
            raise KeyError(f"Name {name} is unavailable")
        return self.df[self.df.key == name].iloc[0].to_dict()

    def get_df(self, name):
        return self.df


class DatasetRegistry:

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Checkpoint path unavailable")
        self.path = path
        self.df = pd.read_csv(os.path.join(path, 'metadata.csv'))
        self.df['available'] = self.df.database.apply(lambda file: os.path.exists(os.path.join(path, file)))
        self.df['available'] &= self.df.target_files.apply(lambda file: os.path.exists(os.path.join(path, file)))
        self.df['available'] &= self.df.list_of_files.apply(lambda file: os.path.exists(os.path.join(path, file)))

    def get_data_list(self):
        return self.df[self.df.is_good & self.df.available].key.tolist()

    def get_data(self, name):
        if not name in self.get_data_list():
            raise KeyError(f"Name {name} is unavailable")
        return self.df[self.df.key == name].iloc[0].to_dict()

    def get_df(self):
        return self.df