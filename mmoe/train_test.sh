#!/bin/bash
set -x
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opt/nvidia_lib
/opt/conda/envs/py27/bin/python multigpu_cnn.py --num_gpus=1  --data_dir=afs/train_data
#sleep 5000
