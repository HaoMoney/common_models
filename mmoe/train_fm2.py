# -*- coding: utf-8 -*-
"""
@author: 
@date: 2020-12-03
@func:
    
"""


from extract_feature import *
from FM import *
import sys
import numpy as np
from utils import batch_generator_single
num_epochs = 1
batch_size = 32
model_path = "./fm_model/"

config = {}
config['lr'] = 0.01
config['batch_size'] = batch_size
config['reg_l1'] = 2e-2
config['reg_l2'] = 0
config['k'] = 40


labels = []
info_list = []
def inp_fn(data):
    """Extract training data.
    @data   : line in training file.
    @return : training data in required format
    """
    scene_ids = list()
    other_feas = list()
    info_list = list()
    labels = list()
    with open(data) as f:
        for line in f:
            line = line.strip().split("###")
            raw = line[0]
            splits = line[1].split("\t")
            if len(splits) != 8:continue
            scene_ids.append(int(splits[0]))
            tmp = []
            for one in splits[1].split(" "):
                tmp.append(float(one))
            for one in splits[2].split(" "):
                tmp.append(float(one))
            for one in splits[3].split(" "):
                tmp.append(float(one))
            for one in splits[4].split(" "):
                tmp.append(float(one))
            for one in splits[5].split(" "):
                tmp.append(float(one))
            tmp.append(float(splits[6]))
            other_feas.append(tmp)
            info_list.append(raw)
            labels.append(int(splits[7]))
        id_feas=make_one_hot(np.array(scene_ids))
        other_feas=np.array(other_feas)
        all_feas=np.concatenate((id_feas,other_feas),axis=1)
        labels = make_one_hot_label(np.array(labels))
        #labels = np.array(labels)
        print(all_feas.shape)
        print(labels.shape)
        return all_feas,labels 
if __name__=="__main__":
    train_feas, train_labels  = inp_fn('../train_data/raw_model_train.final')
    test_feas, test_labels  = inp_fn('../train_data/raw_model_test.final')
    train_freader = batch_generator_single(train_feas, train_labels, batch_size)
    valid_freader = batch_generator_single(test_feas, test_labels, batch_size)
    f = open("pred_results","w")
    with tf.Session() as sess:
        model = FM(config,964)
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        def train_step(fea, label):
            """
            A single training step
            """
            feed_dict = {
                model.X: fea,
                model.y: label,
                model.keep_prob: 1.0
            }
            
            loss,acc,_ = sess.run([model.loss,model.accuracy,model.train_op],  feed_dict=feed_dict)
            return loss,acc
        def dev_step(fea, label):
            """
            A single testing step
            """ 
            feed_dict = {
                model.X: fea,
                model.y: label,
                model.keep_prob: 1.0
            }
            score, loss, acc, = sess.run([model.y_out_prob,model.loss,model.accuracy],  feed_dict)
            return score,loss,acc
        
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = math.ceil(56961/batch_size) 
            for _ in range(num_batches):
                fea,label = next(train_freader)
                loss_tr,acc = train_step(fea,label)
                accuracy_train += acc
                #loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                loss_train += loss_tr
            loss_train /= num_batches
            accuracy_train /= num_batches

            # Testing
            score,loss_test,acc_test = dev_step(test_feas,test_labels) 
            for i in range(len(test_labels)):
                f.write(str(score[i][1])+"\t"+str(test_labels[i][1])+"\n")
            print("loss: {:.5f} acc: {:.5f}, val_loss: {:.5f} val_acc: {:.5f}".format(
                loss_train, accuracy_train, loss_test, acc_test
            ))
            saver.save(sess, model_path+'/model.ckpt')
