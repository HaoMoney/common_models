#!/bin/bash
set -x
alias python82='/home/users/haoqian/paddle_cpu_2.0rc1_py2.7.17/bin/python'
python82 singlegpu_train_etq.py \
            --train_data_dir=./tmp \
            --batch_size=4 \
            --num_batches=20 \
            --num_epochs=1
#sleep 5000
