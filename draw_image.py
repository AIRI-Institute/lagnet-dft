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

import os, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

def process_0(df, name, key_list, desc, legend=True):

    subset = df[df['nn_name'].isin(key_list)]
    plt.figure(figsize=(10,2.5), dpi=150)
    sns.boxplot(subset, x='core_NMAE', y='data_name', hue="Trained on",
                showmeans=True,
                meanprops={'marker':'|','markeredgecolor':'black','markersize':11,'markerfacecolor':'black','markeredgewidth':1.1},
                flierprops={"marker": "d",'markersize':'1'},
                fill=False, linewidth=1.1)
    if not legend:
        plt.legend([],[], frameon=False)
    plt.xlabel('NMAE(%)')
    plt.ylabel(None)
    plt.xticks(np.arange(0, 4.1, 0.25))
    plt.xlim([0,4])
    plt.savefig(name)
    plt.close('all')


def main():
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.6)

    path_to_json = './test2_output'
    data_array = []

    for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
        with open(os.path.join(path_to_json, file_name)) as json_file:
            data = json.load(json_file)
            data_array.append(data)

    df_meta = pd.DataFrame.from_dict(data_array)

    name_dict = {'test2kScaffoldsFull': 'Scaffolds| test',
                 'test2kStructuresFull': 'Structures| test',
                 'test2kConformersFull': 'Conformations| test'}

    order_dict = {'test2kScaffoldsFull': 10,
                  'test2kStructuresFull': 20,
                  'test2kConformersFull': 30}

    list_of_data = []

    for index, row in df_meta.iterrows():
        if row['data_name'] in name_dict:
            local_data = pd.read_csv(os.path.join(path_to_json, row['NMAE_array']), index_col=0)
            local_data['nn_name'] = row['nn_name']
            local_data['data_name'] = name_dict[row['data_name']]
            local_data['order_in_plot'] = order_dict[row['data_name']]
            local_data['test_name'] = row['test_name']
            list_of_data.append(local_data)

    df = pd.concat(list_of_data)
    df['Trained on'] = df['nn_name'].apply(
        lambda x: 'Multiple conf. per mol.' if 'Full' in x else 'Single conf. per mol.')

    df = df.sort_values(['Trained on', 'order_in_plot'], ascending=True)
    df['core_NMAE'] = df['core_NMAE'] * 100
    df['core_NMAE2'] = df['core_NMAE2'] * 100
    #df = df.drop(df[(df.core_NMAE2 > 5) | (df.core_NMAE2 < 0.0000000001)].index).dropna() # In case of numerical disaster:'(


    process_0(df, 'invDeepDFT.png',['invBigT2kFull', 'invBigT2kUnique'],
              'Prediction quality for several data splits\ninvariant DeepDFT, depht=6, features=128', True)
    process_0(df, 'eqvDeepDFT.png',['eqvBigT2kFull', 'eqvBigT2kUnique'],
              'Prediction quality for several data splits\nequivariant DeepDFT, depht=6, features=128', True)
    process_0(df, 'LAGNet.png',['LAGNetBigT2kFull', 'LAGNetBigT2kUnique'],
              'Prediction quality for several data splits\nextended DeepDFT, depht=4, features=128', True)


if __name__=="__main__":
    main()