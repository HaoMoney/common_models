#!/bin/bash
set -x
alias python82='/home/users/haoqian/paddle_cpu_2.0rc1_py2.7.17/bin/python'
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opt/nvidia_lib
#/opt/conda/envs/py27/bin/python singlegpu_train.py \
#                                    --train_data_dir=./train_data \
#                                    --test_data_dir=./test_data \
#                                    --batch_size=1024 \
#                                    --w2v_data_dir=./thirdparty/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5
#/opt/conda/envs/py27/bin/python singlegpu_shared_train.py \
#                                    --train_data_dir=./train_data \
#                                    --test_data_dir=./test_data \
#                                    --batch_size=1024 \
#                                    --w2v_data_dir=./thirdparty/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5

python82 singlegpu_predict_etq_stdin.py \
            --model_dir=./model_5epochs/model.ckpt
#sleep 5000
