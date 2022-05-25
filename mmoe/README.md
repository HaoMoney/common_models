# MMoE

paper:https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

python version:2.7.17 (refer:paddle_cpu_2.0rc1_py2.7.17.tar.gz)
tensorflow version:1.14

Train Dataset Format:
Query Feature: query_fea1\tquery_fea2\t... Prod Feature: prod_fea1\tprod_fea2\t... label1 label2

Begin to train:
python single_gpu_train_etq.py

