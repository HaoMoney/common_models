#!/bin/bash
set -x
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opt/nvidia_lib
/opt/conda/envs/py27/bin/python singlegpu_conv_train.py --num_gpus=1  --train_data_dir=./train_data --test_data_dir=./test_data
#sleep 5000
