#!/bin/bash
set -x
alias python82='/home/users/haoqian/paddle_cpu_2.0rc1_py2.7.17/bin/python'
python82 singlegpu_conv_predict.py \
            --train_data_dir=../train_data \
            --test_data_dir=../eval_data \
            --batch_size=256 \
            --model_dir=../conv_shared_model/model_2.ckpt
#sleep 5000
