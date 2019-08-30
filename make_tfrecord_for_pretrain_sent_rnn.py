#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-08-02
# @Author : KangYu
# @File   : make_data_for_pretrain_sent_rnn2.py

import os
import pickle
import tensorflow as tf
import concurrent.futures
import jieba
import re
import numpy as np
import concurrent.futures

data_path = "/workspace/speaker_verification/data/pretrain_text"
tfrecord_path = "/workspace/speaker_verification/data/tfrecord"
w2i = pickle.load(open('/share/kangyu/speaker_verification/data/w2i.pkl', 'rb'))
datas = os.listdir(data_path)
window_size = 200
window_step = 100
sent_max_len = 100


def sent2idx(sent, w2i, sent_max_len):
    word = [x for x in list(jieba.cut(sent)) if len(re.sub(r'[^\w]|_', '', x)) > 0]
    sent_len = len(word)
    # take care: we put 0 as default idx
    word_idx = [w2i.get(x, -1) + 1 for x in word]
    if sent_len > sent_max_len:
        word_idx = word_idx[:sent_max_len]
    else:
        word_idx = word_idx + [0] * (sent_max_len - sent_len)
    word_idx = np.array(word_idx)
    return word_idx, np.min([sent_max_len, sent_len])


def do(course_id):
    course_file = open(os.path.join(data_path, course_id), 'r')
    course_content = course_file.readlines()
    course_file.close()
    course_content = [line.strip() for line in course_content if line.strip()]

    if len(course_content) == 0:
        return
    course_id = int(course_id)
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path, "{}.tfrecord".format(course_id)))
    window_num = (len(course_content) - window_size) // window_step
    window_num = window_num + 1 if window_num * window_step + window_size < len(course_content) else window_num
    cnt = 0
    for i in range(window_num + 1):
        # get the position
        window_content = course_content[i * window_step:i * window_step + window_size]
        window_sent_word_idx = []
        window_sent_word_num = []
        window_sent_label = []
        window_content_str = []
        for line in window_content:
            label, sent_text = line.split("\t")
            label = int(label)
            wav_file = "{}_{}.wav".format(course_id, cnt)
            if len(re.sub('[^\w]|_', '', sent_text)) == 0:
                continue
            word_idx, sent_len = sent2idx(sent_text, w2i, sent_max_len)
            window_sent_word_idx.append(word_idx)
            window_sent_word_num.append(sent_len)
            window_sent_label.append(label)
            # add the string into result
            content_str = '{}_{}_{}'.format(wav_file, label, sent_text)
            content_str = content_str.encode()
            window_content_str.append(content_str)
            cnt += 1
        window_length = len(window_sent_label)
        window_sent_word_idx = np.array(window_sent_word_idx).astype('int32')  # [sent_num, sent_max_len]
        features = {}
        features['sent_word_idx'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[window_sent_word_idx.tostring()]))
        features['sent_word_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=window_sent_word_idx.shape))
        features['sent_word_num'] = tf.train.Feature(int64_list=tf.train.Int64List(value=window_sent_word_num))
        features['sent_label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=window_sent_label))
        features['length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[window_length]))
        features['content'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=window_content_str))
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
    writer.close()
    print(course_id, "done tfrecord.")
    return

if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        todo = []
        for course_id in datas:
            future = executor.submit(do, course_id)
            todo.append(future)

