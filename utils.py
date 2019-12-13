# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr
import random


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def get_emb(vec_dir):
    with open(vec_dir,'r',encoding='utf-8') as vecs:
        emb = []
        for line in vecs:
            row = line.strip().split(' ')
            emb.append(row)
        emb = np.asarray(emb, dtype = "float32")
        return emb


def read_vocab(vocab_dir):
    word_to_id = {}
    words = []
    with open(vocab_dir,'r',encoding='utf-8') as vocabs:
        i = 0
        for line in vocabs:
            line = line.strip('\n')
            words.append(line)
            word_to_id[line] = i
            i += 1
    return words,word_to_id


def process_file_1(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def process_file_weibo(filename,word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        #print(contents)
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(labels[i])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=2)
    return x_pad, y_pad

def process_file(filename,word_to_id,words, max_length=40):
    """将文件转换为id表示"""
    data_id, label_id = [], []
    words = set(words)
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            if len(line.split('\t')) < 2:
                continue
            label,sent = line.split('\t')
            sent_id = [word_to_id[w] for w in sent.split() if w in words ]
            data_id.append(sent_id)
            label_id.append(label)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=2)
    return x_pad, y_pad

def process_file_trans(filename,word_to_id,words, max_length=40):
    """将文件转换为id表示"""
    data_id, label_id = [], []
    words = set(words)
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            label,sent = line.split('\t')
            sent_id = [word_to_id[w] for w in sent.split() if w in words ]
            data_id.append(sent_id)
            label_id.append(label)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=2)
    return x_pad, y_pad

def process_data_circle(data,word_to_id,words, max_length=40):
    """将文件转换为id表示"""
    data_id, label_id = [], []
    words = set(words)
    for line in data:
        label,sent = line.split('\t')
        sent_id = [word_to_id[w] for w in sent.split() if w in words ]
        data_id.append(sent_id)
        label_id.append(label)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=4)
    return x_pad, y_pad

def process_file_yt_circle(filename,word_to_id,words, max_length=40):
    """
    将文件转换为id表示
    解决数据不均衡的问题
    """
    data_set = []   # 保存完成的数据
    data_id, label_id = [], []
    words = set(words)  # 生成词表索引
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    with open(filename,'r',encoding='utf-8') as f:
        data = f.readlines()
        random.shuffle(data)    # 打乱顺序
        for line in data:
            label,sent = line.split('\t')
            #  print(label+"\t"+sent)
            if label == "0":
                c0.append(line)
            elif label == "1":
                c1.append(line)
            elif label == "2":
                c2.append(line)
            elif label == "3":
                c3.append(line)
            else:
                pass
        # 获取数据组数
        length = int((len(c3)+len(c2)+len(c1))/3)
        sets_num = int(len(c0)/length)
        # 组合数据
        for i in range(sets_num):
            start = i * length
            end = min(( i + 1 ) * length, len(c0))
            print("start:",start,"end:",end)
            s = c0[start : end ]
            s.extend(c1)
            s.extend(c2)
            s.extend(c3)

            data_set.append(s)  # 保存到大集合中去
            # print(data_set)
    return data_set 

def process_file_yt_vote(filename,word_to_id,words, max_length=40):
    """
    将文件转换为id表示
    解决数据不均衡的问题
    """
    data_set = []   # 保存完成的数据
    data_id, label_id = [], []
    words = set(words)  # 生成词表索引
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    with open(filename,'r',encoding='utf-8') as f:
        data = f.readlines()
        random.shuffle(data)    # 打乱顺序
        for line in data:
            label,sent = line.split('\t')
            #  print(label+"\t"+sent)
            if label == "0":
                c0.append(line)
            elif label == "1":
                c1.append(line)
            elif label == "2":
                c2.append(line)
            elif label == "3":
                c3.append(line)
            else:
                pass
        # 获取数据组数
        length = int((len(c3)+len(c2)+len(c1))*5)
        sets_num = int(len(c0)/length)
        print("sets_num:",sets_num)
        # 组合数据
        for i in range(sets_num):
            start = i * length
            end = min(( i + 1 ) * length, len(c0))
            print("start:",start,"end:",end)
            s = c0[start : end ]
            s.extend(c1)
            s.extend(c2)
            s.extend(c3)

            data_set.append(s)  # 保存到大集合中去
            # print(data_set)
    return data_set 

def process_file_yt_online(filename,word_to_id,words, max_length=40):
    """将文件转换为id表示"""
    data_id = []
    words = set(words)
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            sent = line
            sent_id = [word_to_id[w] for w in sent.split() if w in words ]
            data_id.append(sent_id)
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad


def process_file_yt_online_one(sent,word_to_id,words, max_length=40):
    """将文件转换为id表示"""
    data_id = []
    words = set(words)
    sent_id = [word_to_id[w] for w in sent.split() if w in words ]
    data_id.append(sent_id)
    print(sent_id)
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad


def get_result_1(filename,result):
    with open(filename,'r',encoding='utf-8') as f:
        i = 0 
        data = []
        change_to_chinese = {0:'正常',1:'违法',2:'色情',3:'涉政'}
        #change_back = {0:1,1:2,2:3,3:5}
        for line in f:
            line =str( result[i]) + ',' + change_to_chinese[result[i]]  +',' + line
            data.append(line)
            i += 1
        return data 

def get_result(filename,result):
    with open(filename,'r',encoding='utf-8') as f:
        i = 0 
        data = []
        change_to_chinese = {0:'正常',1:'违法',2:'色情',3:'涉政'}
        change_back = {0:1,1:2,2:3,3:5}
        for line in f:
            line =str( result[i]) + ',' +  str(change_back[result[i]]) + ',' + change_to_chinese[result[i]]+ ','  + line
            data.append(line)
            i += 1
        return data 

def get_result(filename,result):
    with open(filename,'r',encoding='utf-8') as f:
        i = 0 
        data = []
        change_to_chinese = {0:'正常',1:'负面'}
        #change_back = {0:1,1:2,2:3,3:5}
        for line in f:
            line =str( result[i]) + ',' + change_to_chinese[result[i]]+ ','  + line
            data.append(line)
            i += 1
        return data 

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]



#a, b = read_category()
#print(a)
#print(b)
