#!/bin/bash

# Choose limit of samples to turn on partial evaluation of metrics
LIMIT_SAMPLES=10000000000 # 2

# Choose CPU or GPU computation
RUNSCRIPT='python3 ./main.py --config-name article_test2.yaml test.accelerator=cpu test.device=cpu'
#RUNSCRIPT='python3 ./main.py --config-name article_test2.yaml test.accelerator=gpu test.device=cuda'

# Evaluation of checkpoints
${RUNSCRIPT} data_key='test2kConformersFull' checkpoint_key='invBigT2kFull' nn=big_DeepDFTinv test.dataset_subset_size=${LIMIT_SAMPLES}
${RUNSCRIPT} data_key='test2kConformersFull' checkpoint_key='eqvBigT2kFull' nn=big_DeepDFTeqv test.dataset_subset_size=${LIMIT_SAMPLES}
${RUNSCRIPT} data_key='test2kConformersFull' checkpoint_key='LAGNetBigT2kFull' nn=big_LAGNet test.dataset_subset_size=${LIMIT_SAMPLES}
${RUNSCRIPT} data_key='test2kConformersFull' checkpoint_key='invBigT2kUnique' nn=big_DeepDFTinv test.dataset_subset_size=${LIMIT_SAMPLES}
${RUNSCRIPT} data_key='test2kConformersFull' checkpoint_key='eqvBigT2kUnique' nn=big_DeepDFTeqv test.dataset_subset_size=${LIMIT_SAMPLES}
${RUNSCRIPT} data_key='test2kConformersFull' checkpoint_key='LAGNetBigT2kUnique' nn=big_LAGNet test.dataset_subset_size=${LIMIT_SAMPLES}

# Draw image
python3 ./draw_image.py
