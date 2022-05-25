# -*- coding: utf-8 -*-
"""
@author: 
@date: 2021-05-10
@func:
    
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import sys

def cal_auc(label, pred, user_id=None):
    '''
    :param label: ground truth
    :param prob: predicted prob
    :param user_id: user index
    :return: gauc
    '''
    if user_id:
        if(len(label) != len(user_id)):
            raise ValueError("impression id num should equal to the sample num,"\
                             "impression id num is {}".format(len(user_id)))
        group_truth = defaultdict(lambda: [])
        group_score = defaultdict(lambda: [])
        for idx, truth in enumerate(label):
            uid = user_id[idx]
            group_truth[uid].append(label[idx])
            group_score[uid].append(pred[idx])
        group_flag = defaultdict(lambda: False)
        for uid in set(user_id):
            truths = group_truth[uid]
            for i in range(len(truths)-1):
                if(truths[i] != truths[i+1]):
                    flag = True
                    break
            group_flag[uid] = flag
        total_auc = 0
        total_impression = 0
        for uid in group_flag:
            if group_flag[uid]:
                total_auc += len(group_truth[uid]) * roc_auc_score(np.asarray(group_truth[uid]), np.asarray(group_score[uid]))
                total_impression += len(group_truth[uid])
        group_auc = float(total_auc) / total_impression
        group_auc = round(group_auc, 4)
        auc = roc_auc_score(label, pred)
        return group_auc,auc
    else:
        auc = roc_auc_score(label, pred)
        return auc

if __name__ == '__main__':
    b2c_label = []
    b2b_label = []
    zsjm_label = []
    b2c_pred = []
    b2b_pred = []
    zsjm_pred = []
    b2c = ["1355", "1519", "1520", "1521", "1526", "1528", "1529"]
    b2b = ["1353", "1356", "927", "993", "994", "1565", "1358", "1359", "1623", "1624"]
    zsjm = ["1607", "1608", "1609", "1610"]
    for line in sys.stdin:
        line = line.strip().split("\t")
        if len(line) < 11:
            continue
        cmatch = line[2]
        ctr_score = float(line[-1])
        ctr_label = int(line[-2])
        if cmatch in b2c:
            b2c_label.append(ctr_label)
            b2c_pred.append(ctr_score)
        elif cmatch in b2b:
            b2b_label.append(ctr_label)
            b2b_pred.append(ctr_score)
        elif cmatch in zsjm:
            zsjm_label.append(ctr_label)
            zsjm_pred.append(ctr_score)
        
    b2c_auc = cal_auc(b2c_label, b2c_pred)
    b2b_auc = cal_auc(b2b_label, b2b_pred)
    zsjm_auc = cal_auc(zsjm_label, zsjm_pred)
    print("b2c_auc: {:.4f}".format(b2c_auc))
    print("b2b_auc: {:.4f}".format(b2b_auc))
    print("zsjm_auc: {:.4f}".format(zsjm_auc))
