#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
##                  tf1.14 单机作业示例                   ##
##                 请将下面的 ak/sk 替换成自己的 ak/sk              ##
###############################################################
cur_time=`date  +"%Y%m%d%H%M"`
job_name=haoqian_tf_siamese_job${cur_time}

# 作业参数
group_name="cmrd-32g-0-yq01-k8s-gpu-v100-8"                   # 将作业提交到group_name指定的组，必填
job_version='tensorflow-1.14.0'

start_cmd="sh train_multi.sh"
wall_time="10:00:00"
k8s_priority="high"
file_dir="."
# 你的ak/sk（可在paddlecloud web页面【个人中心】处获取）
ak="732b8a195a8e568bacfc1506cb12585e"
sk="89822c9c4e665b40b8577ea6741e8a03"

paddlecloud job --ak ${ak} --sk ${sk} \
        train --job-name ${job_name} \
        --group-name ${group_name} \
        --job-conf config.ini \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --k8s-trainers 1 \
        --k8s-gpu-cards 2 \
        --is-standalone 1 \
