from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.layers.python.layers.layers import batch_norm,convolution2d,max_pool2d,fully_connected,flatten
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.metrics import roc_curve, auc, precision_recall_curve
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
_datasets = ['wgEncodeAwgTfbsSydhK562Arid3asc8821IggrabUniPk',
'wgEncodeAwgTfbsSydhK562Atf106325UniPk',
'wgEncodeAwgTfbsHaibK562Atf3V0416101UniPk',
'wgEncodeAwgTfbsSydhK562Bach1sc14700IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Bcl3Pcr1xUniPk',
'wgEncodeAwgTfbsHaibK562Bclaf101388Pcr1xUniPk',
'wgEncodeAwgTfbsSydhK562Bdp1UniPk',
'wgEncodeAwgTfbsSydhK562Bhlhe40nb100IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Cbx3sc101004V0422111UniPk',
'wgEncodeAwgTfbsSydhK562Ccnt2UniPk',
'wgEncodeAwgTfbsSydhK562CebpbIggrabUniPk',
'wgEncodeAwgTfbsHaibK562Cebpbsc150V0422111UniPk',
'wgEncodeAwgTfbsBroadK562Chd1a301218aUniPk',
'wgEncodeAwgTfbsSydhK562Chd2ab68301IggrabUniPk',
'wgEncodeAwgTfbsHaibK562CtcfcPcr1xUniPk',
'wgEncodeAwgTfbsUwK562CtcfUniPk',
'wgEncodeAwgTfbsSydhK562CtcfbIggrabUniPk',
'wgEncodeAwgTfbsBroadK562CtcfUniPk',
'wgEncodeAwgTfbsUtaK562CtcfUniPk',
'wgEncodeAwgTfbsHaibK562Ctcflsc98982V0416101UniPk',
'wgEncodeAwgTfbsSydhK562E2f4UcdUniPk',
'wgEncodeAwgTfbsHaibK562E2f6V0416102UniPk',
'wgEncodeAwgTfbsSydhK562E2f6UcdUniPk',
'wgEncodeAwgTfbsHaibK562Egr1V0416101UniPk',
'wgEncodeAwgTfbsHaibK562Elf1sc631V0416102UniPk',
'wgEncodeAwgTfbsSydhK562P300IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Ets1V0416101UniPk',
'wgEncodeAwgTfbsSydhK562CfosUniPk',
'wgEncodeAwgTfbsUchicagoK562EfosUniPk',
'wgEncodeAwgTfbsHaibK562Fosl1sc183V0416101UniPk',
'wgEncodeAwgTfbsHaibK562GabpV0416101UniPk',
'wgEncodeAwgTfbsSydhK562Gata1UcdUniPk',
'wgEncodeAwgTfbsUchicagoK562Egata2UniPk',
'wgEncodeAwgTfbsHaibK562Gata2sc267Pcr1xUniPk',
'wgEncodeAwgTfbsSydhK562Gata2UcdUniPk',
'wgEncodeAwgTfbsSydhK562Gtf2bUniPk',
'wgEncodeAwgTfbsBroadK562Hdac1sc6298UniPk',
'wgEncodeAwgTfbsHaibK562Hdac2sc6296V0416102UniPk',
'wgEncodeAwgTfbsBroadK562Hdac2a300705aUniPk',
'wgEncodeAwgTfbsBroadK562Hdac6a301341aUniPk',
'wgEncodeAwgTfbsSydhK562Hmgn3UniPk',
'wgEncodeAwgTfbsSydhK562CjunUniPk',
'wgEncodeAwgTfbsUchicagoK562EjunbUniPk',
'wgEncodeAwgTfbsUchicagoK562EjundUniPk',
'wgEncodeAwgTfbsSydhK562JundIggrabUniPk',
'wgEncodeAwgTfbsSydhK562Kap1UcdUniPk',
'wgEncodeAwgTfbsBroadK562Plu1UniPk',
'wgEncodeAwgTfbsSydhK562MaffIggrabUniPk',
'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
'wgEncodeAwgTfbsSydhK562MaxIggrabUniPk',
'wgEncodeAwgTfbsHaibK562MaxV0416102UniPk',
'wgEncodeAwgTfbsSydhK562Mazab85725IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Mef2aV0416101UniPk',
'wgEncodeAwgTfbsSydhK562Mxi1af4185IggrabUniPk',
'wgEncodeAwgTfbsSydhK562CmycIggrabUniPk',
'wgEncodeAwgTfbsUtaK562CmycUniPk',
'wgEncodeAwgTfbsSydhK562Nfe2UniPk',
'wgEncodeAwgTfbsSydhK562NfyaUniPk',
'wgEncodeAwgTfbsSydhK562NfybUniPk',
'wgEncodeAwgTfbsHaibK562Nr2f2sc271940V0422111UniPk',
'wgEncodeAwgTfbsSydhK562Nrf1IggrabUniPk',
'wgEncodeAwgTfbsBroadK562Phf8a301772aUniPk',
'wgEncodeAwgTfbsHaibK562Pmlsc71910V0422111UniPk',
'wgEncodeAwgTfbsBroadK562Pol2bUniPk',
'wgEncodeAwgTfbsHaibK562Pol24h8V0416101UniPk',
'wgEncodeAwgTfbsSydhK562Pol2UniPk',
'wgEncodeAwgTfbsSydhK562Pol2IggmusUniPk',
'wgEncodeAwgTfbsHaibK562Pol2V0416101UniPk',
'wgEncodeAwgTfbsUtaK562Pol2UniPk',
'wgEncodeAwgTfbsSydhK562Pol2s2IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Rad21V0416102UniPk',
'wgEncodeAwgTfbsBroadK562Rbbp5a300109aUniPk',
'wgEncodeAwgTfbsSydhK562Corestsc30189IggrabUniPk',
'wgEncodeAwgTfbsSydhK562Corestab24166IggrabUniPk',
'wgEncodeAwgTfbsHaibK562NrsfV0416102UniPk',
'wgEncodeAwgTfbsSydhK562Rfx5IggrabUniPk',
'wgEncodeAwgTfbsBroadK562Sap3039731UniPk',
'wgEncodeAwgTfbsSydhK562Setdb1UcdUniPk',
'wgEncodeAwgTfbsHaibK562Sin3ak20V0416101UniPk',
'wgEncodeAwgTfbsSydhK562Sirt6UniPk',
'wgEncodeAwgTfbsHaibK562Six5Pcr1xUniPk',
'wgEncodeAwgTfbsSydhK562Smc3ab9263IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Sp1Pcr1xUniPk',
'wgEncodeAwgTfbsHaibK562Sp2sc643V0416102UniPk',
'wgEncodeAwgTfbsHaibK562Pu1Pcr1xUniPk',
'wgEncodeAwgTfbsHaibK562SrfV0416101UniPk',
'wgEncodeAwgTfbsHaibK562Stat5asc74442V0422111UniPk',
'wgEncodeAwgTfbsHaibK562Taf1V0416101UniPk',
'wgEncodeAwgTfbsHaibK562Taf7sc101167V0416101UniPk',
'wgEncodeAwgTfbsSydhK562Tal1sc12984IggmusUniPk',
'wgEncodeAwgTfbsSydhK562Tblr1ab24550IggrabUniPk',
'wgEncodeAwgTfbsSydhK562Tblr1nb600270IggrabUniPk',
'wgEncodeAwgTfbsSydhK562TbpIggmusUniPk',
'wgEncodeAwgTfbsHaibK562Tead4sc101184V0422111UniPk',
'wgEncodeAwgTfbsHaibK562Thap1sc98174V0416101UniPk',
'wgEncodeAwgTfbsHaibK562Trim28sc81411V0422111UniPk',
'wgEncodeAwgTfbsSydhK562Ubfsc13125IggmusUniPk',
'wgEncodeAwgTfbsSydhK562Ubtfsab1404509IggmusUniPk',
'wgEncodeAwgTfbsHaibK562Usf1V0416101UniPk',
'wgEncodeAwgTfbsSydhK562Usf2IggrabUniPk',
'wgEncodeAwgTfbsHaibK562Yy1V0416102UniPk',
'wgEncodeAwgTfbsHaibK562Yy1V0416101UniPk',
'wgEncodeAwgTfbsHaibK562Zbtb33Pcr1xUniPk',
'wgEncodeAwgTfbsHaibK562Zbtb7asc34508V0416101UniPk',
'wgEncodeAwgTfbsSydhK562Znf143IggrabUniPk',
'wgEncodeAwgTfbsSydhK562Znf263UcdUniPk',
'wgEncodeAwgTfbsSydhK562Znf274m01UcdUniPk',
'wgEncodeAwgTfbsSydhK562Znf274UcdUniPk']

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

def load_ENCODE_k562_dataset(dataset_num, motif_occ):
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
        if motif_occ==1:
            full_filename = '../data/cnn.csail.mit.edu/motif_occupancy/' + dataset_name + '/' + fname
        else:
            full_filename = '../data/cnn.csail.mit.edu/motif_discovery/' + dataset_name + '/' + fname
        f = open(full_filename)
        num_lines = 0
        for line in f:
            match = re.search("chr([0123456789]+):(.*)-(.*) ([ATGC]+) ([01])", line)
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
            match = re.search("chr([0123456789]+):(.*)-(.*) ([ATGC]+) ([01])", line)
            if match:
                chr_num = int(match.group(1)) # the chromosome number (eg. '1' for chr1)
                #left_idx = int(match.group(2)) # the left index
                #right_idx = int(match.group(3)) # the right index
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
    validation_size = int(train_features.shape[0] * 0) # percent of training features to use as validation set; 0 to disable validation set
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
    f.write('dataset_num,motif_discovery=0|motif_occupancy=1,dataset_name,roc_auc,prc_auc,time(sec)\n')
    f.close()
    for dataset_num in range(0, len(_datasets)):
        for motif_occ in range(0,2):
            success = False
            try:
                load_ENCODE_k562_dataset(dataset_num,motif_occ)
                success = True
            except:
                print('Hmm.. Something happened. Skipping dataset ' + _datasets[dataset_num])
            if success:
                with tf.variable_scope('scopename_' + str(dataset_num) + '_' + str(motif_occ)):
                    # LSTM Parameters ============================
                    lstm_n_hidden = 32 # hidden layer num features
                    # ============================================

                    x = tf.placeholder(tf.float32, shape=[None, 101*4])
                    y_ = tf.placeholder(tf.float32, shape=[None, 2])

                    # Create the model
                    x_image = tf.reshape(x, [-1,101,4,1])

                    # CONVOLUTIONAL LAYER(S)
                    n_conv1 = 384
                    L_conv1 = 9
                    maxpool_len1 = 2
                    conv1 = convolution2d(x_image, n_conv1, [L_conv1,4], padding='VALID', normalizer_fn=None)
                    conv1_pool = max_pool2d(conv1, [maxpool_len1,1], [maxpool_len1,1])
                    #conv1_drop = tf.nn.dropout(conv1_pool, conv_keep_prob)
                    conv1_pool_len = int((101-L_conv1+1)/maxpool_len1)

                    n_conv2 = n_conv1
                    L_conv2 = 5
                    maxpool_len2 = int(conv1_pool_len-L_conv2+1) # global maxpooling (max-pool across temporal domain)
                    conv2 = convolution2d(conv1_pool, n_conv2, [L_conv2,1], padding='VALID', normalizer_fn=None)
                    conv2_pool = max_pool2d(conv2, [maxpool_len2,1], [maxpool_len2,1])
                    #conv2_drop = tf.nn.dropout(conv2_pool, conv_keep_prob)

                    # LINEAR FC LAYER
                    y_conv = fully_connected(flatten(conv2_pool), 2, activation_fn=None)
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
                        #if i%100 == 0:
                        if 1 == 0: # turned off
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
                        test_batch = get_next_batch(2, 64)
                        if pred_test_labels is None:
                            pred_test_labels = y_conv_softmax.eval(feed_dict={x: test_batch[0], y_: test_batch[1]})
                            true_test_labels = test_batch[1]
                        else:
                            pred_test_labels = numpy.vstack([pred_test_labels, y_conv_softmax.eval(feed_dict={x: test_batch[0], y_: test_batch[1]})])
                            true_test_labels = numpy.vstack([true_test_labels, test_batch[1]])
                    fpr, tpr, _ = roc_curve(true_test_labels[:,0], pred_test_labels[:,0]) # get receiver operating characteristics
                    precision, recall, _ = precision_recall_curve(true_test_labels[:,0], pred_test_labels[:,0]) # get precision recall curve
                    roc_auc = auc(fpr, tpr)
                    prc_auc = auc(recall, precision)
                    print("%s, dataset %g, final test roc auc %g, final test prc auc %g, time elapsed %g seconds"%(_datasets[dataset_num], dataset_num, roc_auc, prc_auc, time.time()-t0))
                    f=open(file_name,'a')
                    f.write(str(dataset_num) + ',' + str(motif_occ) + ',' + _datasets[dataset_num] + ',' + str(roc_auc) + ',' + str(prc_auc) + ',' + str(time.time()-t0) + '\n')
                    f.close()
                    t0 = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()