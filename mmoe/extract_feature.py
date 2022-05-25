#encoding:utf-8
import sys
import numpy as np
def extract_feature():
    fo1 = open('train.feature','w')
    fo2 = open('words_list','w')
    w2id = dict()
    for item in sys.stdin:
        tmp = []
        item = item.strip()
        item = item.split('\t')
        if len(item) < 3:continue
        label = item[0]
        tags = item[2].split('$')
        for tag in tags:
            if tag not in w2id:
                wid = len(w2id) + 100
                w2id[tag] = str(wid)
                tmp.append(str(wid))
            else:
                wid = w2id[tag]
                tmp.append(wid)
        fo1.write(label + '\t' + ' '.join(tmp) + ' ' + '\n')
    
    for item in w2id:
        fo2.write(item + '\t' + w2id[item] + '\n')
    fo1.close()
    fo2.close()

def make_one_hot(data1,depth):
    return (np.arange(depth)==data1[:,None]).astype(np.integer)

def load_data(data_file):
    """
    """
    fi = open(data_file)
    y = []
    X = []
    for line in fi:
        splits = line.strip().split('\t')
        if len(splits) < 2:continue
        y.append(int(splits[0]) - 1)
        X.append([int(i) for i in splits[1].split(' ')])
    X_train = np.array(X[0:4200000])
    y_train = np.array(y[0:4200000])
    X_test = np.array(X[4200000:])
    y_test = np.array(y[4200000:])
    y_test = make_one_hot(y_test)
    y_train = make_one_hot(y_train)
    return X_train, y_train, X_test, y_test

def load_test_data(data_file):
    """
    """
    fi = open(data_file)
    y = []
    X = []
    for line in fi:
        splits = line.strip().split('\t')
        y.append(splits[0])
        X.append([int(i) for i in splits[1].split(' ')]) 
    X_train = np.array(X[0:140000])
    y_train = np.array(y[0:140000])
    return X_train, y_train

def pretrain_extract_feature():
    """
    """
    fo1 = open('../data_0604/train_pretrain.feature','w')
    w2id = dict()
    with open('../data_0527/pretrain.words_list.id') as f:
        datas = f.readlines()
        for item in datas:
            item = item.strip()
            if item:
                item = item.split('\t')
                if len(item) < 2:
                    continue
                w2id[item[0]] = item[1]

    for item in sys.stdin:
        tmp = []
        item = item.strip()
        item = item.split('\t')
        if len(item) < 3:continue
        label = item[0]
        tags = item[2].split('$')
        for tag in tags:
            if tag in w2id:
                id = w2id[tag]
                tmp.append(id)
        if len(tmp) > 0:
            fo1.write(label + '\t' + ' '.join(tmp) + ' ' + '\n')

if __name__ == "__main__":
    c = [[1],[2],[3]]
    print(make_one_hot(c))
    #extract_feature()
    #pretrain_extract_feature()
