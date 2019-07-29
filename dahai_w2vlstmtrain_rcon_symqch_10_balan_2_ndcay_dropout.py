import re
import os
import sys
import numpy as np
import tensorflow as tf

def parse_helper(example_proto):
    dics = {'voice_embed': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'voice_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'sent_word_idx': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'sent_word_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'sent_word_num': tf.VarLenFeature(dtype=tf.int64),
            'sent_label': tf.VarLenFeature(dtype=tf.int64),
            'length': tf.FixedLenFeature(shape=(), dtype=tf.int64)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    sent_word_num = tf.sparse_tensor_to_dense(parsed_example['sent_word_num'])
    sent_label = tf.sparse_tensor_to_dense(parsed_example['sent_label'])
    sent_word_num = tf.cast(sent_word_num, tf.int32)
    sent_label = tf.cast(sent_label, tf.int32)
    voice_embed = tf.decode_raw(parsed_example['voice_embed'], tf.float32)
    voice_embed = tf.reshape(voice_embed, parsed_example['voice_shape'])
    sent_word_idx = tf.decode_raw(parsed_example['sent_word_idx'], tf.int32)
    sent_word_idx = tf.reshape(sent_word_idx, parsed_example['sent_word_shape'])
    length = tf.cast(parsed_example['length'],tf.int32)
    return voice_embed, sent_word_idx, sent_word_num, sent_label, length

def optim(lr, config):
    """ return optimizer determined by configuration
    :return: tf optimizer
    """
    if config.optim == "sgd":
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim == "rmsprop":
        return tf.train.RMSPropOptimizer(lr)
    elif config.optim == "adam":
        return tf.train.AdamOptimizer(lr)
    else:
        raise AssertionError("Wrong optimizer type!")

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

def train_attention(config):
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.99

    # draw the graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, None)
    voice_embed = tf.placeholder(shape=[config.batch_size, config.max_sent_num, 64], dtype=tf.float32) # shape: (batch, max_sent_num, 64)

    # Build the netword for the words
    word_idx = tf.placeholder(shape=[config.batch_size, config.max_sent_num, config.max_sent_len], dtype=tf.int32) # shape: (batch, max_sent_num, max_sent_len)
    sent_len = tf.placeholder(shape=[config.batch_size, config.max_sent_num], dtype=tf.int32) # shape: (batch, max_sent_num)
    w2v_embedding = tf.Variable(tf.constant(0.0, shape=[config.vocab_size, config.w2v_dim]),trainable=config.w2v_istrain, name="w2i_embedding")
    embedding_placeholder = tf.placeholder(tf.float32, [config.vocab_size, config.w2v_dim])
    embedding_init = w2v_embedding.assign(embedding_placeholder)

    word_embed = tf.nn.embedding_lookup(w2v_embedding, word_idx) # shape: (batch, max_sent_num, max_sent_len, 200)

    word_embed = tf.reshape(word_embed, shape=[-1,config.max_sent_len, config.w2v_dim]) # [batch*max_sent_num, max_sent_len, 200]
    word_embed_len = tf.reshape(sent_len, [-1]) # [batch*max_sent_num]
    with tf.variable_scope("sent_embed_lstm"):
        # add the dropout
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=128) for i in range(1)]
        # add drop-out op
        lstm_cells = [tf.contrib.rnn.DropoutWrapper(x,input_keep_prob=1.0-config.drop_rate) for x in lstm_cells]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # define lstm op and variables
        _, states = tf.nn.dynamic_rnn(cell=lstm, inputs=word_embed, dtype=tf.float32, time_major=False, sequence_length=word_embed_len) # [layer_num,2,batch*max_sent_num,lstm_proj_dim]
        word_embed = states[-1][1] # [batch*max_sent_num,lstm_proj_dim]
        word_embed = tf.reshape(word_embed, shape=[config.batch_size, -1, 128]) # [batch, max_sent_num, lstm_proj_dim]

    label = tf.placeholder(shape=[config.batch_size, config.max_sent_num], dtype=tf.int32) # shape: (batch, max_sent_num)
    seq_len = tf.placeholder(shape=[config.batch_size], dtype=tf.int32) # shape: (batch)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)    
    
    voice_Q = tf.layers.dense(inputs=voice_embed, units=32, activation=None, use_bias=False,
                        kernel_initializer=tf.truncated_normal_initializer(),name='attention_w')
    voice_K = tf.layers.dense(inputs=voice_embed, units=32, activation=None, use_bias=False, 
                        name='attention_w', reuse=True)
    word_V = word_embed # shape: (batch, max_length, 200)
    voice_A = tf.matmul(voice_Q,voice_K,transpose_b=True) # shape: (batch, max_length, max_length)
    voice_A = tf.transpose(voice_A, [0,2,1]) # transpose to get the correct format
    voice_A = mask(voice_A, seq_len, mode='add') # shape: (batch, max_length, max_length)
    voice_A = tf.transpose(voice_A, [0,2,1]) # transpose to get the correct format
    voice_A = tf.nn.softmax(voice_A) # shape: (batch, max_length, max_length)
    # add drop-out op
    voice_A = tf.layers.dropout(voice_A, rate=config.drop_rate, training=is_training)
    voice_O = tf.matmul(voice_A,word_V) # shape: (batch, max_length, 200)
    voice_O = mask(voice_O, seq_len, mode='mul', max_len=config.max_sent_num) # shape: (batch, max_sent_num, 200)

    O = tf.concat([voice_O, word_V], axis = -1)
    # add another full connected layer
    dense1 = tf.layers.dense(inputs=O, units=128, activation=tf.nn.relu, 
                                kernel_initializer=tf.truncated_normal_initializer(),
                                bias_initializer=tf.constant_initializer(0.1)) # shape: (batch, max_length, 32)
    dense1 = tf.layers.dropout(dense1, rate=config.drop_rate, training=is_training)
    logits= tf.layers.dense(inputs=dense1, units=2, activation=None, 
                            kernel_initializer=tf.truncated_normal_initializer(),
                            bias_initializer=tf.constant_initializer(0.1)) # shape: (batch, max_length, 3)    

    # calculate the loss
    label_onehot = tf.one_hot(indices=label, depth=2, on_value=1, off_value=0, axis=-1, dtype=tf.int32)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(label_onehot, shape=[-1,2]), logits=tf.reshape(logits,shape=[-1,2])) # shape: (batch*max_length,1)
    loss = tf.reshape(loss, shape=[config.batch_size, -1]) # [batch_size, max_length]
    # add a mask based on whether it is 0
    balance_mask_1 = tf.cast(tf.sequence_mask(seq_len, config.max_sent_num),tf.float32)
    balance_mask_2 = tf.cast(tf.equal(label,1), tf.float32)*1
    balance_mask_3 = tf.cast(tf.equal(label,0), tf.float32)*2
    temp_mask = tf.cast(balance_mask_2>0, tf.float32) + tf.cast(balance_mask_3>0, tf.float32)
    balance_mask = balance_mask_1 * (balance_mask_2 + balance_mask_3)
    loss = loss * balance_mask
    loss = tf.reduce_sum(loss, axis=1) # shape: (batch)
    
    temp_label = tf.expand_dims(label, -1) # [batch, max_length, 1]
    temp_label = tf.matmul(1-temp_label,temp_label, transpose_b=True) + tf.matmul(temp_label, 1-temp_label, transpose_b=True) # [batch, max_len, max_len]
    constraint_mask_1 = tf.cast(tf.equal(temp_label,1),tf.float32) # [batch, max_len, max_len]
    constraint_mask_2 = tf.cast(tf.sequence_mask(seq_len, config.max_sent_num),tf.float32) # [batch, max_length]
    constraint_mask_2 = tf.expand_dims(constraint_mask_2, -1) # [batch, max_length, 1]
    constraint_mask_2 = tf.matmul(constraint_mask_2, constraint_mask_2, transpose_b=True) # [batch, max_length, max_length]
    constraint_mask = constraint_mask_1 * constraint_mask_2
    # care about the attentions
    attention_loss = voice_A**2 * constraint_mask # [batch, max_length, max_length]
    attention_loss = tf.reduce_sum(attention_loss, axis=[1,2]) # [batch]

    loss = loss + config.alpha*attention_loss

    loss = tf.reduce_mean(loss/tf.cast(tf.reduce_sum(balance_mask, axis=1), dtype=tf.float32))
    # add the accuracy
    pred_label = tf.cast(tf.argmax(logits,-1),tf.int32)
    correct_label = tf.cast(tf.equal(pred_label, label),tf.float32)
    correct_label = correct_label * balance_mask_1 * temp_mask
    accuracy = tf.cast(tf.reduce_sum(correct_label),tf.float32)/tf.cast(tf.reduce_sum(temp_mask*balance_mask_1),tf.float32)

    base_acc = tf.cast(tf.equal(label,1),tf.float32) # [batch_size, max_sent]
    base_acc = base_acc * balance_mask_1 * temp_mask
    base_acc = tf.cast(tf.reduce_sum(base_acc),tf.float32)/tf.reduce_sum(tf.cast(tf.reduce_sum(temp_mask*balance_mask_1),tf.float32))

    # then we will define the trainable variables
    trainable_vars= tf.trainable_variables()                # get variable list
    optimizer= optim(lr, config)                                    # get optimizer (type is determined by configuration)
    grads, vars= zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss
    train_op= optimizer.apply_gradients(zip(grads, vars), global_step= global_step)   # gradient update operation
    # check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)
    loss_summary = tf.summary.scalar("Loss", loss)
    acc_summary = tf.summary.scalar("Accuracy", accuracy)
    base_acc_summary = tf.summary.scalar("Base_Accuracy", base_acc)
    merge_summary = tf.summary.merge([loss_summary, acc_summary, base_acc_summary])

    w2v_np = np.load('/workspace/HangLi/ocean_talk/w2v_online/0718_dahaipretrain/w2v_mtrx.npy')
    w2v_np = np.concatenate([np.array([[0.0]*config.w2v_dim]),w2v_np],axis=0)

    train_root = '/workspace/HangLi/ocean_talk/dahai/0622_600/va_widxdahai_tfrecord_200_100/'
    train_dataset = tf.data.TFRecordDataset([os.path.join(train_root,x) for x in os.listdir(train_root)])
    parsed_train = train_dataset.map(parse_helper)
    parsed_train = parsed_train.shuffle(10000)
    parsed_train = parsed_train.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=config.batch_size,padded_shapes=(
                [config.max_sent_num,64],[config.max_sent_num,config.max_sent_len],
                [config.max_sent_num],[config.max_sent_num],[])))
    parsed_train = parsed_train.repeat()
    train_iter = parsed_train.make_one_shot_iterator()
    train_next = train_iter.get_next()

    test_zhikang_root = '/workspace/HangLi/ocean_talk/zhikang/yeqiu_clip_0429/va_widxdahai_tfrecord_200_200/'
    test_zhikang_dataset = tf.data.TFRecordDataset([os.path.join(test_zhikang_root,x) for x in os.listdir(test_zhikang_root)])
    parsed_test_zhikang = test_zhikang_dataset.map(parse_helper)
    parsed_test_zhikang = parsed_test_zhikang.shuffle(10000)
    parsed_test_zhikang = parsed_test_zhikang.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=config.batch_size,padded_shapes=(
                [config.max_sent_num,64],[config.max_sent_num,config.max_sent_len],
                [config.max_sent_num],[config.max_sent_num],[])))
    parsed_test_zhikang = parsed_test_zhikang.repeat()
    test_zhikang_iter = parsed_test_zhikang.make_one_shot_iterator()
    test_zhikang_next = test_zhikang_iter.get_next()

    test_dahai_root = '/workspace/HangLi/ocean_talk/dahai/test_0527/va_widxdahai_tfrecord_200_200/'
    test_dahai_dataset = tf.data.TFRecordDataset([os.path.join(test_dahai_root,x) for x in os.listdir(test_dahai_root)])
    parsed_test_dahai = test_dahai_dataset.map(parse_helper)
    parsed_test_dahai = parsed_test_dahai.shuffle(10000)
    parsed_test_dahai = parsed_test_dahai.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=config.batch_size,padded_shapes=(
                [config.max_sent_num,64],[config.max_sent_num,config.max_sent_len],
                [config.max_sent_num],[config.max_sent_num],[])))
    parsed_test_dahai = parsed_test_dahai.repeat()
    test_dahai_iter = parsed_test_dahai.make_one_shot_iterator()
    test_dahai_next = test_dahai_iter.get_next()

    with tf.Session(config = gpu_config) as sess:
        tf.global_variables_initializer().run()
        # also initialize the embedding weight
        sess.run(embedding_init, feed_dict={embedding_placeholder:w2v_np})
        saver = tf.train.Saver()
        os.makedirs(os.path.join(config.model_path, "Check_Point"), exist_ok=True)  # make folder to save model
        os.makedirs(os.path.join(config.model_path, "logs"), exist_ok=True)          # make folder to save log
        train_writer = tf.summary.FileWriter(os.path.join(config.model_path, "logs/train"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(config.model_path, "logs/test_zhikang"), sess.graph)
        test_writer_2 = tf.summary.FileWriter(os.path.join(config.model_path, "logs/test_dahai"), sess.graph)
        lr_factor = 1   # lr decay factor ( 1/2 per 10000 iteration)
        loss_acc, test_loss_acc, test_loss_acc_2 = 0, 0, 0
        acc_acc, test_acc_acc , test_acc_acc_2= 0, 0, 0
        base_acc_acc, test_base_acc_acc, test_base_acc_acc_2 = 0, 0, 0
        for iter in range(config.iteration):
            # run forward and backward propagation and update parameters
            train_voice_embed, train_word_idx, train_sent_len, train_label, train_seq_len = sess.run(train_next)
            test_voice_embed, test_word_idx, test_sent_len, test_label, test_seq_len = sess.run(test_zhikang_next)
            test_voice_embed_2, test_word_idx_2, test_sent_len_2, test_label_2, test_seq_len_2 = sess.run(test_dahai_next)

            _, loss_cur, train_summary, acc_cur, base_acc_cur = sess.run([train_op, loss, merge_summary, accuracy, base_acc],
                                                                            feed_dict={voice_embed: train_voice_embed, word_idx:train_word_idx, sent_len: train_sent_len,
                                                                                    label: train_label, seq_len: train_seq_len, lr: config.lr*lr_factor,is_training:True})
            test_loss_cur, test_summary, test_acc_cur, test_base_acc_cur = sess.run([loss, merge_summary, accuracy, base_acc],
                                                                                    feed_dict={voice_embed: test_voice_embed, word_idx:test_word_idx, sent_len: test_sent_len,
                                                                                                label: test_label, seq_len: test_seq_len, is_training:False})
            test_loss_cur_2, test_summary_2, test_acc_cur_2, test_base_acc_cur_2 = sess.run([loss, merge_summary, accuracy, base_acc],
                                                                                            feed_dict={voice_embed: test_voice_embed_2, word_idx:test_word_idx_2, sent_len: test_sent_len_2,
                                                                                                        label: test_label_2, seq_len: test_seq_len_2, is_training:False})
            loss_acc += loss_cur    # accumulated loss for each 100 iteration
            test_loss_acc += test_loss_cur
            test_loss_acc_2 += test_loss_cur_2
            acc_acc += acc_cur
            test_acc_acc += test_acc_cur
            test_acc_acc_2 += test_acc_cur_2
            base_acc_acc += base_acc_cur
            test_base_acc_acc += test_base_acc_cur
            test_base_acc_acc_2 += test_base_acc_cur_2

            if iter % 10 == 0:
                train_writer.add_summary(train_summary, iter)   # write at tensorboard
                test_writer.add_summary(test_summary, iter)
                test_writer_2.add_summary(test_summary_2, iter)

            if (iter+1) % 100 == 0:
                print("(iter : %d)\ntrain_loss: %.4f/train_acc: %.4f/base_acc: %.4f\ntest_zhikang_loss: %.4f/test_zhikang_acc: %.4f/base_acc: %.4f\ntest_dahai_loss: %.4f/test_dahai_acc: %.4f/base_acc: %.4f" % (
                    (iter+1),loss_acc/100,acc_acc/100,base_acc_acc/100,test_loss_acc/100,test_acc_acc/100,test_base_acc_acc/100,test_loss_acc_2/100,test_acc_acc_2/100,test_base_acc_acc_2/100))
                loss_acc, test_loss_acc, test_loss_acc_2 = 0, 0, 0
                acc_acc, test_acc_acc , test_acc_acc_2= 0, 0, 0
                base_acc_acc, test_base_acc_acc, test_base_acc_acc_2 = 0, 0, 0

            if (iter+1) % config.lr_decay_step == 0:
                lr_factor /= 2                      # lr decay
                print("learning rate is decayed! current lr : ", config.lr*lr_factor)

            if (iter+1) % config.model_save_step == 0:
                saver.save(sess, os.path.join(config.model_path, "./Check_Point/model.ckpt"), global_step=iter//config.model_save_step)
                print("model is saved!")

class configuration(object):
    def __init__(self):
        return None

if __name__ == "__main__":
    config = configuration()
    config.batch_size = 32
    config.optim = 'sgd'
    config.iteration = 50000
    config.lr = 0.1
    config.drop_rate = 0.5
    config.model_path = '/workspace/HangLi/ocean_talk/Voice_Attention/model_w2vin_dahai/w2vlstmtrain_rcon_symqch_10_balan_2_ndcay_dropout/'
    config.lr_decay_step = 50000
    config.model_save_step = 10000
    config.alpha = 10
    config.max_sent_num = None
    config.max_sent_len=100
    config.w2v_dim = 200
    config.vocab_size = 314041
    config.w2v_istrain = True

    train_attention(config)