from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.layers.python.layers.layers import batch_norm,convolution2d,max_pool2d,fully_connected,flatten
from sklearn.metrics import roc_curve, auc
import numpy
import re
import random
import math
import time

import tensorflow as tf

numpy.set_printoptions(threshold=numpy.nan) # print all contents when printing numpy arrays (good for debugging)

sess = tf.InteractiveSession()

FLAGS = None

# Global variables:
_index_in_epoch_train = 0
_index_in_epoch_test = 0
_index_in_epoch_validation = 0
_train_features = None
_train_labels = None
_validation_features = None
_validation_labels = None
_test_features = None
_test_labels = None
_train_epochs_completed = 0
_validation_epochs_completed = 0
_test_epochs_completed = 0
_validation_size = 0
_datasets = ['wgEncodeAwgTfbsBroadK562CtcfUniPk',
    'wgEncodeAwgTfbsHaibK562Atf3V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Cebpbsc150V0422111UniPk',
    'wgEncodeAwgTfbsHaibK562CtcfcPcr1xUniPk',
    'wgEncodeAwgTfbsHaibK562E2f6V0416102UniPk',
    'wgEncodeAwgTfbsHaibK562Egr1V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Elf1sc631V0416102UniPk',
    'wgEncodeAwgTfbsHaibK562Ets1V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Fosl1sc183V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Gata2sc267Pcr1xUniPk',
    'wgEncodeAwgTfbsHaibK562MaxV0416102UniPk',
    'wgEncodeAwgTfbsHaibK562Mef2aV0416101UniPk',
    'wgEncodeAwgTfbsHaibK562NrsfV0416102UniPk',
    'wgEncodeAwgTfbsHaibK562Pu1Pcr1xUniPk',
    'wgEncodeAwgTfbsHaibK562Sp1Pcr1xUniPk',
    'wgEncodeAwgTfbsHaibK562Sp2sc643V0416102UniPk',
    'wgEncodeAwgTfbsHaibK562SrfV0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Stat5asc74442V0422111UniPk',
    'wgEncodeAwgTfbsHaibK562Tead4sc101184V0422111UniPk',
    'wgEncodeAwgTfbsHaibK562Thap1sc98174V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Usf1V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Yy1V0416101UniPk',
    'wgEncodeAwgTfbsHaibK562Yy1V0416102UniPk',
    'wgEncodeAwgTfbsHaibK562Zbtb33Pcr1xUniPk',
    'wgEncodeAwgTfbsHaibK562Zbtb7asc34508V0416101UniPk',
    'wgEncodeAwgTfbsSydhK562Arid3asc8821IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Atf106325UniPk',
    'wgEncodeAwgTfbsSydhK562Bach1sc14700IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Bhlhe40nb100IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562CebpbIggrabUniPk',
    'wgEncodeAwgTfbsSydhK562CfosUniPk',
    'wgEncodeAwgTfbsSydhK562CjunUniPk',
    'wgEncodeAwgTfbsSydhK562CmycIggrabUniPk',
    'wgEncodeAwgTfbsSydhK562CtcfbIggrabUniPk',
    'wgEncodeAwgTfbsSydhK562E2f4UcdUniPk',
    'wgEncodeAwgTfbsSydhK562E2f6UcdUniPk',
    'wgEncodeAwgTfbsSydhK562Gata1UcdUniPk',
    'wgEncodeAwgTfbsSydhK562Gata2UcdUniPk',
    'wgEncodeAwgTfbsSydhK562JundIggrabUniPk',
    'wgEncodeAwgTfbsSydhK562MaffIggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562MaxIggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Nfe2UniPk',
    'wgEncodeAwgTfbsSydhK562NfyaUniPk',
    'wgEncodeAwgTfbsSydhK562NfybUniPk',
    'wgEncodeAwgTfbsSydhK562Nrf1IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Rfx5IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562TbpIggmusUniPk',
    'wgEncodeAwgTfbsSydhK562Usf2IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Znf143IggrabUniPk',
    'wgEncodeAwgTfbsSydhK562Znf263UcdUniPk',
    'wgEncodeAwgTfbsUchicagoK562EfosUniPk',
    'wgEncodeAwgTfbsUchicagoK562Egata2UniPk',
    'wgEncodeAwgTfbsUchicagoK562EjunbUniPk',
    'wgEncodeAwgTfbsUchicagoK562EjundUniPk',
    'wgEncodeAwgTfbsUtaK562CmycUniPk',
    'wgEncodeAwgTfbsUtaK562CtcfUniPk',
    'wgEncodeAwgTfbsUwK562CtcfUniPk']

def get_next_batch(dataset_to_use=0, batch_size=128):
    # dataset_to_use value: 0==train, 1==validation, 2==test
    global _index_in_epoch_train
    global _index_in_epoch_test
    global _index_in_epoch_validation
    global _train_features
    global _train_labels
    global _validation_features
    global _validation_labels
    global _test_features
    global _test_labels
    global _train_epochs_completed
    global _validation_epochs_completed
    global _test_epochs_completed
    if dataset_to_use == 0:
        index_in_epoch = _index_in_epoch_train
        num_examples = _train_features.shape[0]
    elif dataset_to_use == 1:
        index_in_epoch = _index_in_epoch_validation
        num_examples = _validation_features.shape[0]
    elif dataset_to_use == 2:
        index_in_epoch = _index_in_epoch_test
        num_examples = _test_features.shape[0]
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_examples:
        if dataset_to_use == 0:
            _train_epochs_completed += 1
        elif dataset_to_use == 1:
            _validation_epochs_completed += 1
        else:
            _test_epochs_completed += 1
        # Shuffle the data
        perm = numpy.arange(num_examples)
        numpy.random.shuffle(perm)
        if dataset_to_use == 0:
            _train_features = _train_features[perm]
            _train_labels = _train_labels[perm]
        elif dataset_to_use == 1:
            _validation_features = _validation_features[perm]
            _validation_labels = _validation_labels[perm]
        elif dataset_to_use == 2:
            _test_features = _test_features[perm]
            _test_labels = _test_labels[perm]
        # Start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    if dataset_to_use == 0:
        _index_in_epoch_train = index_in_epoch
        return _train_features[start:end], _train_labels[start:end]
    elif dataset_to_use == 1:
        _index_in_epoch_validation = index_in_epoch
        return _validation_features[start:end], _validation_labels[start:end]
    else:
        _index_in_epoch_test = index_in_epoch
        return _test_features[start:end], _test_labels[start:end]

def load_ENCODE_k562_dataset(dataset_num):
    global _index_in_epoch_train    
    global _index_in_epoch_test
    global _index_in_epoch_validation
    global _train_features
    global _train_labels
    global _validation_features
    global _validation_labels
    global _test_features
    global _test_labels
    global _train_epochs_completed
    global _validation_epochs_completed
    global _test_epochs_completed
    global _datasets
    global _validation_size
    dataset_name = _datasets[dataset_num]
    test_features = None
    test_labels = None
    train_features = None
    train_labels = None
    for get_test in range(0,2):
        if get_test:
            fname = 'test.data'
        else:
            fname = 'train.data'
        full_filename = '../data/cnn.csail.mit.edu/motif_occupancy/' + dataset_name + '/' + fname
        f = open(full_filename)
        num_lines = 0
        for line in f:
            match = re.search("chr([0123456789]+):([0123456789]*)-([0123456789]*) ([ATGC]+) ([01])", line)
            if match:
                dna_string = match.group(4) # the sequence of DNA base pairs in range
                if len(dna_string) == 101:
                    num_lines = num_lines + 1 # increase counter ONLY if a match & dna string length exactly 101 (bug fixed)
        f.close()
        num_bases = 0
        nparr_features = None
        nparr_labels = None
        cur_line = 0
        f = open(full_filename)
        for line in f:
            match = re.search("chr([0123456789]+):([0123456789]*)-([0123456789]*) ([ATGC]+) ([01])", line)
            if match:
                chr_num = int(match.group(1)) # the chromosome number (eg. '1' for chr1)
                left_idx = int(match.group(2)) # the left index
                right_idx = int(match.group(3)) # the right index
                dna_string = match.group(4) # the sequence of DNA base pairs in range
                bound = int(match.group(5)) # whether or not the transcription factor did bind to this DNA sequence (1 if bound, 0 if not bound)
                if len(dna_string) == 101:
                    if num_bases < 1:
                        num_bases = len(dna_string) * 4
                    if nparr_features is None:
                        nparr_features = numpy.empty([num_lines, num_bases], dtype=numpy.dtype('float32'))
                        nparr_labels = numpy.empty([num_lines, 2], dtype=numpy.dtype('float32'))
                    cur_base = 0
                    for dna_base in dna_string: # one-hot encode the DNA bases:
                        if dna_base is 'A':
                            nparr_features[cur_line, cur_base + 0] = 1.0
                            nparr_features[cur_line, cur_base + 1] = 0.0
                            nparr_features[cur_line, cur_base + 2] = 0.0
                            nparr_features[cur_line, cur_base + 3] = 0.0
                        elif dna_base is 'T':
                            nparr_features[cur_line, cur_base + 0] = 0.0
                            nparr_features[cur_line, cur_base + 1] = 1.0
                            nparr_features[cur_line, cur_base + 2] = 0.0
                            nparr_features[cur_line, cur_base + 3] = 0.0
                        elif dna_base is 'G':
                            nparr_features[cur_line, cur_base + 0] = 0.0
                            nparr_features[cur_line, cur_base + 1] = 0.0
                            nparr_features[cur_line, cur_base + 2] = 1.0
                            nparr_features[cur_line, cur_base + 3] = 0.0
                        else: # everything else is binned to 'C'
                            nparr_features[cur_line, cur_base + 0] = 0.0
                            nparr_features[cur_line, cur_base + 1] = 0.0
                            nparr_features[cur_line, cur_base + 2] = 0.0
                            nparr_features[cur_line, cur_base + 3] = 1.0
                        cur_base = cur_base + 4
                    if bound == 1:
                        nparr_labels[cur_line, 0] = 0.0
                        nparr_labels[cur_line, 1] = 1.0
                    else: # everything else is classed as not-bound
                        nparr_labels[cur_line, 0] = 1.0
                        nparr_labels[cur_line, 1] = 0.0
                    cur_line = cur_line + 1 # inc only if matched & dna string length was exactly 101 (bug fixed)
        if get_test:
            test_features = nparr_features
            test_labels = nparr_labels
        else:
            train_features = nparr_features
            train_labels = nparr_labels
    validation_size = int(train_features.shape[0] * 0.15) # percent of training features to use as validation set; 0 to disable validation set
    _validation_features = train_features[:validation_size]
    _validation_labels = train_labels[:validation_size]
    _train_features = train_features[validation_size:]
    _train_labels = train_labels[validation_size:]
    _test_features = test_features
    _test_labels = test_labels
    _train_epochs_completed = 0
    _validation_epochs_completed = 0
    _test_epochs_completed = 0
    _validation_size = validation_size

def main(_):
    global _train_epochs_completed
    global _validation_epochs_completed
    global _test_epochs_completed
    global _datasets
    global _validation_size
    global _test_labels

    dropout_on = tf.placeholder(tf.float32)
    if dropout_on is not None:
        conv_keep_prob = 1.0
    else:
        conv_keep_prob = 1.0

    file_name = 'out_' + str(int(time.time())) + '.csv'
    f=open(file_name,'w') # clear file
    f.write('dataset_num,dataset_name,roc_auc\n')
    f.close()
    for dataset_num in range(0, len(_datasets)):
        load_ENCODE_k562_dataset(dataset_num)

        x = tf.placeholder(tf.float32, shape=[None, 101*4])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # Create the model
        x_image = tf.reshape(x, [-1,101,4,1])

        # CONVOLUTIONAL LAYER(S)
        n_conv1 = 64
        L_conv1 = 9
        maxpool_len1 = 2
        conv1 = convolution2d(x_image, n_conv1, [L_conv1,4], padding='VALID', normalizer_fn=None)
        conv1_pool = max_pool2d(conv1, [maxpool_len1,1], [maxpool_len1,1])
        conv1_pool_len = int((101-L_conv1+1)/maxpool_len1)

        n_conv2 = n_conv1
        L_conv2 = 5
        maxpool_len2 = 2
        conv2 = convolution2d(conv1_pool, n_conv2, [L_conv2,1], padding='VALID', normalizer_fn=None)
        conv2_pool = max_pool2d(conv2, [maxpool_len2,1], [maxpool_len2,1])
        conv2_pool_len = int((conv1_pool_len-L_conv2+1)/maxpool_len2)

        n_conv3 = n_conv1
        L_conv3 = 3
        maxpool_len3 = 2
        conv3 = convolution2d(conv2_pool, n_conv3, [L_conv3,1], padding='VALID', normalizer_fn=None)
        conv3_pool = max_pool2d(conv3, [maxpool_len3,1], [maxpool_len3,1])
        conv3_pool_len = int((conv2_pool_len-L_conv3+1)/maxpool_len3)

        n_conv4 = n_conv1
        L_conv4 = 3
        maxpool_len4 = int(conv3_pool_len-L_conv3+1) # global maxpooling (across temporal domain)
        conv4 = convolution2d(conv3_pool, n_conv4, [L_conv4,1], padding='VALID', normalizer_fn=None)
        conv4_pool = max_pool2d(conv4, [maxpool_len4,1], [maxpool_len4,1])

        # LINEAR FC LAYER
        y_conv = fully_connected(flatten(conv4_pool), 2, activation_fn=None)
        y_conv_softmax = tf.nn.softmax(y_conv)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.initialize_all_variables())
        
        i = 0
        prev_auc = 0.0001 # small value to prevent DIV0
        stop_condition = None
        t0 = time.time()
        while stop_condition is None:
            if i%1000 == 0:
                #t0 = time.time()
                pred_validation_labels = None
                true_validation_labels = None
                prev_validation_epochs_completed = _validation_epochs_completed
                while _validation_epochs_completed - prev_validation_epochs_completed == 0: # do in mini batches because single GTX970 has insufficient memory to test all at once
                    if _validation_size > 1024*5:
                        validation_batch = get_next_batch(1,1024)
                    else:
                        validation_batch = get_next_batch(1,64)
                    if pred_validation_labels is None:
                        pred_validation_labels = y_conv_softmax.eval(feed_dict={x: validation_batch[0], y_: validation_batch[1]})
                        true_validation_labels = validation_batch[1]
                    else:
                        pred_validation_labels = numpy.vstack([pred_validation_labels, y_conv_softmax.eval(feed_dict={x: validation_batch[0], y_: validation_batch[1]})])
                        true_validation_labels = numpy.vstack([true_validation_labels, validation_batch[1]])
                fpr, tpr, _ = roc_curve(true_validation_labels[:,0], pred_validation_labels[:,0])
                roc_auc = auc(fpr, tpr)
                #check stop condition:
                perc_chg_auc = (roc_auc - prev_auc) / prev_auc
                #if perc_chg_auc < 0.005: # stop when auc moving average on validation set changes by <0.5%
                #    stop_condition = 1
                prev_auc = roc_auc
                print("%s, dataset %g, epoch %d, step %d, time elapsed %g, validation roc auc %g, perc chg in auc %g"%(_datasets[dataset_num], dataset_num, _train_epochs_completed, i, time.time()-t0, roc_auc, perc_chg_auc))
                t0 = time.time()
            batch = get_next_batch(0)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], dropout_on: 1})
            if i == 7000:
                stop_condition = 1
            i += 1

        pred_test_labels = None
        true_test_labels = None
        while _test_epochs_completed == 0: # do testing in mini batches because single GTX970 has insufficient memory to test all at once
            test_batch = get_next_batch(2,64)
            if pred_test_labels is None:
                pred_test_labels = y_conv_softmax.eval(feed_dict={x: test_batch[0], y_: test_batch[1]})
                true_test_labels = test_batch[1]
            else:
                pred_test_labels = numpy.vstack([pred_test_labels, y_conv_softmax.eval(feed_dict={x: test_batch[0], y_: test_batch[1]})])
                true_test_labels = numpy.vstack([true_test_labels, test_batch[1]])
        fpr, tpr, _ = roc_curve(true_test_labels[:,0], pred_test_labels[:,0])
        roc_auc = auc(fpr, tpr)
        print("%s, dataset %g, final test roc auc %g"%(_datasets[dataset_num], dataset_num, roc_auc))
        f=open(file_name,'a')
        f.write(str(dataset_num) + ',' + _datasets[dataset_num] + ',' + str(roc_auc) + '\n')
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()