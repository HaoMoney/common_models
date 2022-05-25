#!/bin/bash
set -x

DIRNAME=$(cd $(dirname $0); pwd)
export ROOT_PATH="$DIRNAME/../"

export TURING_DF_USER_NAME="haoqian01" #�û�ERP��������
export TURING_DF_TRIGGER_TIME="`date "+%Y-%m-%d %H:%M:%S"`" #��ǰʱ��
export TURING_DF_MISSION_ID="0" #Ĭ��Ϊ0����
export TURING_DF_TASK_ID="0" #Ĭ��Ϊ0����
export TURING_DF_INSTANCE_ID="0" #Ĭ��Ϊ0����
export TURING_DF_USER_CODE="044d8e3333" #�û�ͼ����Կ����turing��ҳ���Ͻǻ�ȡ��������

export AFS_CLIENT="/home/users/haoqian/hadoop-client-kunpeng/hadoop/bin/hadoop"
export TURING_CLIENT="/home/users/haoqian/turing-client/hadoop/bin/hadoop"
export TURING_SUBMIT_HOME="/home/users/haoqian/turing-submit/turing-submit"

export PADDLECLOUD="/home/users/haoqian/.jumbo/bin/paddlecloud"

# ------------------------------------------------------------------------
source ${ROOT_PATH}/../conf/env.online.conf
# ------------------------------------------------------------------------
#date1=$1
#date2=$2
#date1=20210623
#date2=20210622

ds=`date "+%Y%m%d%H%M%S"`


#input_path1="afs://tianqi.afs.baidu.com:9902/app/ecom/brand/haoqian01/alad/cuxuan_rollback_product_useful_info/20210718122240/part-00000"
#input_path="afs://tianqi.afs.baidu.com:9902/app/ecom/brand/haoqian01/alad/etq/test_data"
input_path="turingpriv.alad_task/etq/test_data"
output_path="userpath.relevance_model/test_data_sim_1epoch_new"
#donefile_path="afs://tianqi.afs.baidu.com:9902/app/ecom/brand/haoqian01/alad/filtered_ecom_pid/donefile"
#del_path="afs://tianqi.afs.baidu.com:9902/app/ecom/brand/haoqian01/alad/filtered_ecom_pid/$date2"
#src_path="afs://baihua.afs.baidu.com:9902/user/asgard/goods/scan/20210613"
#dst_path="afs://tianqi.afs.baidu.com:9902/app/ecom/brand/haoqian01/alad/search_product/20210613"

$HADOOP_FS -rmr $output_path 
$HADOOP_STREAMING \
    -input ${input_path} \
    -output ${output_path} \
    -mapper "Python-27/bin/python singlegpu_predict_etq_stdin.py --model_dir=./model_1epoch/model.ckpt" \
    -reducer NONE \
    -file singlegpu_predict_etq_stdin.py \
    -file utils.py \
    -file mmoe_etq.py \
    -file extract_feature.py \
    -file batch_process.py \
    -file cal_gauc.py \
    -file attention.py \
    -cacheArchive ${PYTHONTF_CLIENT}#Python-27 \
    -cacheArchive ${MMOE_MODEL_1EPOCH}#model_1epoch \
    -jobconf mapred.job.name="fz_fc_research_haoqian01_test_$ds" \
    -jobconf mapred.job.map.capacity=1000 \
    -jobconf mapred.map.tasks=2000 \
    -jobconf mapred.job.reduce.capacity=1000 \
    -jobconf mapred.reduce.tasks=200 \
    -jobconf mapred.map.over.capacity.allowed=false \
    -jobconf mapred.reduce.over.capacity.allowed=false \
    -jobconf mapred.textoutputformat.ignoreseparator=true \
    -jobconf abaci.job.base.environment=default \
    -jobconf stream.memory.limit=8000 \
    -jobconf mapred.job.priority=VERY_HIGH

if [[ $? -ne 0 ]]; then
	echo 'Job Failed !!!'
	exit 1
fi

#echo $output_path >price_donefile
#$hadoop_tq -rmr $donefile_path
#$hadoop_tq -put price_donefile $donefile_path
#$hadoop_tq -rmr $del_path 

exit 0
