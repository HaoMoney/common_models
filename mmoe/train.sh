#!/bin/bash
set -x
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

cat ./train_data/* >./all_train_data
/opt/conda/envs/py27/bin/python singlegpu_train_etq.py \
                                    --train_data_dir=./all_train_data \
                                    --batch_size=256 \
                                    --num_epochs=10
#sleep 5000
