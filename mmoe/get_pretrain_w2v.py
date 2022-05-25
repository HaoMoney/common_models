#encoding:utf-8
#author:victory
#date:20180504
#todo:
"""
input: w2v file
return:w2v numpy
 
"""
import numpy as np

def loadVec(filename, w2v_dim):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename,'r')
    word_dim = w2v_dim
    print(w2v_dim)
    embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        if len(row) < 201:continue
        vec = map(float, row[1:])
        embd.append(vec)
    fr.close()
    embedding = np.asarray(embd)
    return embedding

def loadWord2Vec(filename, vocab_size, w2v_dim):
    vocab = []
    #embd = []
    embd = np.zeros((vocab_size+1, w2v_dim))
    cnt = 0
    fr = open(filename,'r')
    #word_dim = w2v_dim    
    #embd.append([0]*word_dim)
    i = 0
    for line in fr :
        row = line.strip().split(' ')
        if len(row) != (w2v_dim+1):continue
        embd[i+1] = np.asarray(row[1:])
        i += 1
    fr.close()
    #embedding = np.asarray(embd)
    return embd
