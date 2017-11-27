import os, sys
import argparse
import time
import itertools
import cPickle
import logging
import random
import string

import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import nottingham_util
import util
from model import Model, NottinghamModel
from rnn import DefaultConfig

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Script to train and save a model.')

    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='softmax',
                        # choices = ['bach', 'nottingham', 'softmax'],
                        choices = ['softmax'])
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--run_name', type=str, default=time.strftime("%m%d_%H%M"))


    args = parser.parse_args()


    if args.dataset == 'softmax':
        time_step = 120
        model_class = NottinghamModel
        with open(nottingham_util.PICKLE_LOC, 'r') as f:
            pickle = cPickle.load(f)
            chord_to_idx = pickle['chord_to_idx']

        input_dim = pickle["train"][0].shape[1]
        print 'Finished loading data, input dim: {}'.format(input_dim)
    else:
        raise Exception("Other datasets not yet implemented")

    # initializer = tf.random_uniform_initializer(-0.1, 0.1)

    best_config = None
    best_valid_loss = None

    # set up run dir
    run_folder = os.path.join(args.model_dir, args.run_name)


    logger = logging.getLogger(__name__) 
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(run_folder, "training.log")))

    with open(args.config_file, 'r') as f: 
        config = cPickle.load(f)
    print(config) #print config and add to log


    if config.dataset == 'softmax':
        data = util.load_data('', time_step, config.time_batch_len, config.max_time_batches, nottingham=pickle)
    else:
        raise Exception("Other datasets not yet implemented")



    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model", reuse=None):
            train_model = model_class(config, training=True)
        with tf.variable_scope("model", reuse=True):
            valid_model = model_class(config, training=False)

        #restore model
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=40)
        saver.restore(session, os.path.join(run_folder, config.model_name))
        
        # training
        early_stop_best_loss = None
        start_saving = False
        saved_flag = False
        train_losses, valid_losses = [], []
        start_time = time.time()
        for i in range(config.num_epochs):

            loss = util.run_epoch(session, train_model, 
                data["train"]["data"], training=True, testing=False)
            train_losses.append((i, loss))
            
            if i == 0:
                continue

            logger.info('Epoch: {}, Train Loss: {}, Time Per Epoch: {}'.format(\
                    i, loss, (time.time() - start_time)/i))

            valid_loss = util.run_epoch(session, valid_model, data["valid"]["data"], training=False, testing=False)
            valid_losses.append((i, valid_loss))
            logger.info('Valid Loss: {}'.format(valid_loss))

            if early_stop_best_loss == None:
                early_stop_best_loss = valid_loss
            elif valid_loss < early_stop_best_loss:
                early_stop_best_loss = valid_loss
                if start_saving:
                    logger.info('Best loss so far encountered, saving model.')
                    saver.save(session, os.path.join(run_folder, config.model_name))
                    saved_flag = True
            elif not start_saving:
                start_saving = True 
                logger.info('Valid loss increased for the first time, will start saving models')
                saver.save(session, os.path.join(run_folder, config.model_name))
                saved_flag = True

        if not saved_flag:
            saver.save(session, os.path.join(run_folder, config.model_name))

        # set loss axis max to 20
        axes = plt.gca()
        if config.dataset == 'softmax':
            axes.set_ylim([0, 2])
        else:
            axes.set_ylim([0, 100])
        plt.plot([t[0] for t in train_losses], [t[1] for t in train_losses])
        plt.plot([t[0] for t in valid_losses], [t[1] for t in valid_losses])
        plt.legend(['Train Loss', 'Validation Loss'])
        chart_file_path = os.path.join(run_folder, 'result.png')
        plt.savefig(chart_file_path)
        plt.clf()

        logger.info("Config {}, Loss: {}".format(config, early_stop_best_loss))
        if best_valid_loss == None or early_stop_best_loss < best_valid_loss:
            logger.info("Found best new model!")
            best_valid_loss = early_stop_best_loss
            best_config = config

    logger.info("Best Config: {}, Loss: {}".format(best_config, best_valid_loss))
