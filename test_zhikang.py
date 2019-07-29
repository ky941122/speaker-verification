#coding=utf-8

import pickle
import pandas as pd
import os
import tensorflow as tf

def format_result(label_result, asr_content):
    label_result_dict = dict([(int(x[0].split('.')[0].split('_')[1]), x[1]) for x in label_result])
    result = [dict(list(x.items()) + [('role',label_result_dict.get(x['sentence_id'],-1))]) for x in asr_content]
    # transform them into the ms format
    result = [(x['role'],x['begin_time'],x['end_time']) for x in result if x['role'] != -1]
    return result

def pred_parse_helper(example_proto):
    dics = {'voice_embed': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'voice_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'sent_word_idx': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'sent_word_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'sent_word_num': tf.VarLenFeature(dtype=tf.int64),
            'sent_label': tf.VarLenFeature(dtype=tf.int64),
            'content': tf.VarLenFeature(dtype=tf.string),
            'length': tf.FixedLenFeature(shape=(), dtype=tf.int64)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    sent_word_num = tf.sparse_tensor_to_dense(parsed_example['sent_word_num'])
    sent_label = tf.sparse_tensor_to_dense(parsed_example['sent_label'])
    content = tf.sparse_tensor_to_dense(parsed_example['content'],default_value='')
    sent_word_num = tf.cast(sent_word_num, tf.int32)
    sent_label = tf.cast(sent_label, tf.int32)
    voice_embed = tf.decode_raw(parsed_example['voice_embed'], tf.float32)
    voice_embed = tf.reshape(voice_embed, parsed_example['voice_shape'])
    sent_word_idx = tf.decode_raw(parsed_example['sent_word_idx'], tf.int32)
    sent_word_idx = tf.reshape(sent_word_idx, parsed_example['sent_word_shape'])
    length = tf.cast(parsed_example['length'],tf.int32)
    return voice_embed, sent_word_idx, sent_word_num, sent_label, content, length

def mask(inputs, seq_len, mode='mul',max_len=None):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len,max_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12
        
def predict_attention(config):
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.99

    # draw the graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, None)
    voice_embed = tf.placeholder(shape=[None, None, 64], dtype=tf.float32) # shape: (batch, max_sent_num, 64)

    # Build the netword for the words
    word_idx = tf.placeholder(shape=[config.batch_size, config.max_sent_num, config.max_sent_len],
                              dtype=tf.int32)  # shape: (batch, max_sent_num, max_sent_len)
    sent_len = tf.placeholder(shape=[config.batch_size, config.max_sent_num],
                              dtype=tf.int32)  # shape: (batch, max_sent_num)
    w2v_embedding = tf.Variable(tf.constant(0.0, shape=[config.vocab_size, config.w2v_dim]),
                                trainable=config.w2v_istrain, name="w2i_embedding")
    embedding_placeholder = tf.placeholder(tf.float32, [config.vocab_size, config.w2v_dim])
    embedding_init = w2v_embedding.assign(embedding_placeholder)

    word_embed = tf.nn.embedding_lookup(w2v_embedding, word_idx)  # shape: (batch, max_sent_num, max_sent_len, 200)
    word_embed = tf.layers.dropout(word_embed, rate=config.drop_rate - 0.3, training=is_training)

    # do the mask based on the max_sent_len
    sent_len_mask = tf.sequence_mask(sent_len, maxlen=config.max_sent_len)  # (batch, max_sent_num, max_sent_len)
    sent_len_mask = tf.cast(sent_len_mask, tf.float32)
    sent_len_mask = tf.expand_dims(sent_len_mask, -1)  # (batch, max_sent_num, max_sent_len, 1)
    # Here is for the mean pooling
    word_embed = word_embed * sent_len_mask
    word_embed = tf.reduce_sum(word_embed, axis=2)  # (batch, max_sent_num, 200)
    sent_len_temp = sent_len + tf.cast(tf.equal(sent_len, 0), tf.int32)
    sent_len_temp = tf.cast(tf.expand_dims(sent_len_temp, -1), tf.float32)  # (batch, max_sent_num, 1)
    word_embed = word_embed / sent_len_temp  # (batch, max_sent_num, 200)
    word_embed = tf.layers.dropout(word_embed, rate=config.drop_rate - 0.3, training=is_training)

    # #####  RNN for words in a sentence #####
    # fw_cell = tf.nn.rnn_cell.GRUCell(config.rnn_units, name="forward_cell")
    # bw_cell = tf.nn.rnn_cell.GRUCell(config.rnn_units, name="backward_cell")
    #
    # word_embed = tf.reshape(word_embed, [-1, config.max_sent_len, 200])  # [batch*max_sent_num, max_sent_len, 200]
    # seq_len4rnn = tf.reshape(sent_len, [-1])
    #
    # rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_embed, sequence_length=seq_len4rnn, dtype=tf.float32)
    # rnn_out = tf.concat(rnn_out, axis=-1)
    #
    # rnn_out = tf.reshape(rnn_out, [config.batch_size, -1, config.max_sent_len, config.rnn_units*2])
    #
    # sent_len_mask = tf.sequence_mask(sent_len, maxlen=config.max_sent_len)  # (batch, max_sent_num, max_sent_len)
    # sent_len_mask = tf.cast(sent_len_mask, tf.float32)
    # sent_len_mask = tf.expand_dims(sent_len_mask, -1)  # (batch, max_sent_num, max_sent_len, 1)
    # # Here is for the mean pooling
    # word_embed = rnn_out * sent_len_mask
    # word_embed = tf.reduce_sum(word_embed, axis=2)  # (batch, max_sent_num, 200)
    # sent_len_temp = sent_len + tf.cast(tf.equal(sent_len, 0), tf.int32)
    # sent_len_temp = tf.cast(tf.expand_dims(sent_len_temp, -1), tf.float32)  # (batch, max_sent_num, 1)
    # word_embed = word_embed / sent_len_temp  # (batch, max_sent_num, 200)
    # word_embed = tf.layers.dropout(word_embed, rate=config.drop_rate - 0.2, training=is_training)

    label = tf.placeholder(shape=[config.batch_size, config.max_sent_num],
                           dtype=tf.int32)  # shape: (batch, max_sent_num)
    seq_len = tf.placeholder(shape=[config.batch_size], dtype=tf.int32)  # shape: (batch)
    lr = tf.placeholder(dtype=tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)

    voice_Q = tf.layers.dense(inputs=voice_embed, units=32, activation=None, use_bias=False,
                              kernel_initializer=tf.truncated_normal_initializer(), name='attention_w')
    voice_K = tf.layers.dense(inputs=voice_embed, units=32, activation=None, use_bias=False,
                              name='attention_w', reuse=True)
    word_V = word_embed  # shape: (batch, max_length, 200)
    voice_A = tf.matmul(voice_Q, voice_K, transpose_b=True)  # shape: (batch, max_length, max_length)
    voice_A = tf.transpose(voice_A, [0, 2, 1])  # transpose to get the correct format
    voice_A = mask(voice_A, seq_len, mode='add')  # shape: (batch, max_length, max_length)
    voice_A = tf.transpose(voice_A, [0, 2, 1])  # transpose to get the correct format
    voice_A = tf.nn.softmax(voice_A)  # shape: (batch, max_length, max_length)

    ####  top k  ####
    _, top_indices = tf.nn.top_k(voice_A, k=config.k, sorted=False)
    top_mask = tf.one_hot(indices=top_indices, depth=config.max_sent_num, dtype=tf.float32)
    top_mask = tf.reduce_sum(top_mask, axis=-2)
    voice_A_top = voice_A * top_mask
    #################

    voice_O = tf.matmul(voice_A_top, word_V)  # shape: (batch, max_length, 200)
    voice_O = mask(voice_O, seq_len, mode='mul', max_len=config.max_sent_num)  # shape: (batch, max_sent_num, 200)

    O = tf.concat([voice_O, word_V], axis=-1)

    # O = tf.keras.layers.BatchNormalization(axis=-1)(O, training=is_training)
    # O = tf.layers.batch_normalization(O, axis=-1, training=is_training)

    ###### RNN for sentences #####

    fw_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=config.rnn_units, use_peepholes=True, name="forward_cell_2")
    bw_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=config.rnn_units, use_peepholes=True, name="backward_cell_2")
    rnn_out_2, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_2, bw_cell_2, O, sequence_length=seq_len,
                                                   dtype=tf.float32)
    rnn_out_2 = tf.concat(rnn_out_2, axis=-1)

    # add another full connected layer
    dense1 = tf.layers.dense(inputs=rnn_out_2, units=128, activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             bias_initializer=tf.constant_initializer(0.1))  # shape: (batch, max_length, 32)
    dense1 = tf.layers.dropout(dense1, rate=config.drop_rate, training=is_training)

    dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             bias_initializer=tf.constant_initializer(0.1))

    logits = tf.layers.dense(inputs=dense2, units=2, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(),
                             bias_initializer=tf.constant_initializer(0.1))  # shape: (batch, max_length, 3)

    pred_label = tf.cast(tf.argmax(logits, -1), tf.int32)

#     pred_root = '/workspace/HangLi/ocean_talk/zhikang/yeqiu_clip_0429/va_widx_tfrecord/'
    pred_root = '/workspace/speaker_verification/data/zhikang/test/va_widxdahai_tfrecord_200_200'
    pred_dataset = tf.data.TFRecordDataset([os.path.join(pred_root,x) for x in os.listdir(pred_root)])
    parsed_pred = pred_dataset.map(pred_parse_helper)
    parsed_pred = parsed_pred.padded_batch(batch_size=config.batch_size,padded_shapes=([None,64],[None,None],
                                                                                       [None],[None],[None],[]))
    pred_iter = parsed_pred.make_one_shot_iterator()
    pred_next = pred_iter.get_next()

    with tf.Session(config=gpu_config) as sess:
        saver = tf.train.Saver(var_list=tf.global_variables())
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(config.model_path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                loaded = 1
                model = os.path.join(config.model_path,'Check_Point',os.path.basename(model))
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        result = []
        result_A = []
        while True:
            try:
                pred_voice_embed, pred_word_idx, pred_sent_len, true_label, pred_content, pred_seq_len = sess.run(pred_next)
            except tf.errors.OutOfRangeError:
                break
            label_result = sess.run(pred_label, feed_dict={voice_embed: pred_voice_embed, word_idx:pred_word_idx, 
                                                           sent_len: pred_sent_len,seq_len: pred_seq_len,is_training:False})
            A_mtrx = sess.run(voice_A, feed_dict={voice_embed:pred_voice_embed, seq_len:pred_seq_len,is_training:False})

            label_result = [x for x in label_result.flat]
            wav_result = ['{}.wav'.format(x.decode().split('.wav_')[0]) for x in pred_content.flat]
            temp_result = [x for x in zip(wav_result, label_result) if x[0]!='.wav']
            result += temp_result
            result_A.append((A_mtrx, true_label, pred_seq_len))
    return result, result_A
  
class configuration(object):
  def __init__(self):
    return

config = configuration()
config.batch_size = 16
config.drop_rate = 0.5
config.model_path = '/workspace/HangLi/ocean_talk/Voice_Attention/model_w2vin_dahai/w2vlstmtrain_rcon_symqch_10_balan_2_ndcay_dropout/'
config.model_num = 4
config.w2v_dim = 200
config.vocab_size = 314041
config.max_sent_len = 100

va_pred_label, va_attention = predict_attention(config)


pred_timeline = []
asr_root = '/workspace/HangLi/ocean_talk/zhikang/guowei_result/'
for course_id in list(set([x[0].split('_')[0] for x in va_pred_label])):
    pred_label_result = [x for x in va_pred_label if x[0].split('_')[0] == course_id]
    asr_content = pd.read_excel(os.path.join(asr_root, '{}.xlsx'.format(course_id)))
    asr_content['sentence_id'] = range(asr_content.shape[0])
    asr_content = asr_content[['begin_time','end_time','sentence_id']].to_dict('records')
    pred_result = format_result(pred_label_result, asr_content)
    pred_timeline.append((int(course_id), pred_result))
pred_timeline = dict(pred_timeline)

pickle.dump(pred_timeline, open('/workspace/HangLi/ocean_talk/zhikang/yeqiu_clip_0429/classification/dahai_w2vlstmtrain_rcon_symqch_10_balan_2_ndcay_dropout.pkl','wb'))