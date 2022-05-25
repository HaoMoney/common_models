# -*- coding: utf-8 -*-
"""
@author: 
@date: 2021-05-10
@func:
    
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

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
    user_id = ['a', 'a', 'a', 'b', 'b', 'b', 'a']
    label = [1, 0, 1, 0, 1, 1, 0]
    pred = [0.4, 0.5, 0.7, 0.2, 0.6, 0.7, 0.4]
    #group_auc = gauc(label, pred, user_id)
    #print('group_auc: {:.4f}'.format(group_auc))
    #auc = roc_auc_score(label, pred)
    #print("auc: {:.4f}".format(auc))
    auc = cal_auc(label, pred)
    print("auc: {:.4f}".format(auc))
