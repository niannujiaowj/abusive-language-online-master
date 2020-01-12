#coding=utf-8
#!/usr/bin/env python
import numpy as np
from numpy.random import choice
import os
import pandas as pd
import sys
import nltk
import tensorflow as tf
import time

HOME = '.'
sys.path.insert(0, HOME)
from alo import dataset
from alo import word_embed
from alo import nn
from alo import utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_string('corpus', 'rt.txt', 'input corpus')
tf.app.flags.DEFINE_integer('num_filter', 100, 'number of feature maps')
tf.app.flags.DEFINE_float('keep_prob', 0.7, 'dropout keep rate')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('max_len', 124, 'max sentence length')
tf.app.flags.DEFINE_integer('epoches', 32, 'number of epoches')
tf.app.flags.DEFINE_string('embed_fn', 'text8-vector.bin', 'embeddings file')
tf.app.flags.DEFINE_integer('embed_dim', 200, 'word embedding dimentionality')
tf.app.flags.DEFINE_integer('n_tasks', 1, 'number of tasks')
tf.app.flags.DEFINE_integer('task_idx', 1, 'index of the task, 1 for toxicity, 2 for aggression 3 for attack')
tf.app.flags.DEFINE_string('output', 'Output_sep_task_all_b64_x20.txt', 'output_file')
tf.app.flags.DEFINE_integer('xfold', 10, 'number of folds')
tf.app.flags.DEFINE_string('save_to', 'saved_models', 'folder for saved models')


def main(_):
    #wiki = dataset.WikiTalk()
    #data = wiki.data2matrix()
    print('CNN - n_task={}'.format(FLAGS.n_tasks))
    data = np.load(os.path.join(HOME, 'data.npy'))
    output_file = sys.stderr #open(FLAGS.output,"w")
    if FLAGS.n_tasks == 1:
        X_data, Y_data = data[:, :-3], data[:, -FLAGS.task_idx]
    else:
        X_data, Y_data = data[:, :-3], data[:, -3:]
        
    print(X_data.shape)
    print(Y_data.shape)

    embed_path = os.path.join(HOME, 'resource', FLAGS.embed_fn)
    embeddings = word_embed.Word2Vec(embed_path)
    word_embeddings = np.array(list(embeddings.word2embed.values()))

    dir_saved_models = os.path.join(HOME, FLAGS.save_to)
    if not os.path.exists(dir_saved_models):
        os.makedirs(dir_saved_models)

    fold_size = X_data.shape[0] // FLAGS.xfold
    with tf.Graph().as_default(), tf.Session() as sess:
        for i in range(FLAGS.xfold):
            if i != 2:  # fold 3 yields the best result
                continue
            
            print('{}\nValidating fold {}, validate_idx = [{}:{}]\n{}'.format(
                '-'*79, i+1, i*fold_size, (i+1)*fold_size, '-'*79), file=output_file, flush=True)

            X_train = np.vstack((X_data[:i*fold_size], X_data[(i+1)*fold_size:]))
            if FLAGS.n_tasks == 1:
                Y_train = np.hstack((Y_data[:i*fold_size], Y_data[(i+1)*fold_size:]))
            else:
                Y_train = np.vstack((Y_data[:i*fold_size], Y_data[(i+1)*fold_size:]))
            X_test = X_data[i*fold_size:(i+1)*fold_size]
            Y_test = Y_data[i*fold_size:(i+1)*fold_size]

            if FLAGS.n_tasks == 1:
                tccnn = nn.STCNN2(
                    filter_heights=[3, 4, 5],
                    word_embeddings=word_embeddings,
                    sent_len=FLAGS.max_len,
                    batch_size=FLAGS.batch_size,
                    num_filter=FLAGS.num_filter,
                    keep_prob=FLAGS.keep_prob,
                    lr=FLAGS.lr,
                    embed_dim=FLAGS.embed_dim)
                best_accuracy = np.array([-1.])
            else:
                tccnn = nn.MTCNN2(
                    filter_heights=[3, 4, 5],
                    word_embeddings=word_embeddings,
                    sent_len=FLAGS.max_len,
                    batch_size=FLAGS.batch_size,
                    num_filter=FLAGS.num_filter,
                    keep_prob=FLAGS.keep_prob,
                    lr=FLAGS.lr,
                    embed_dim=FLAGS.embed_dim,
                    n_tasks=FLAGS.n_tasks)
                best_accuracy = np.array([-1., -1., -1.])

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()

            fold_idx = i
            
            # init loss
            train_loss, train_acc = .0, np.array([0.] * FLAGS.n_tasks)
            num_batches = X_train.shape[0] // FLAGS.batch_size
            shuffled_batches = choice(range(num_batches), num_batches, replace=False)
            for _n, idx in enumerate(shuffled_batches):
                X_train_batch = X_train[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                Y_train_batch = Y_train[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                feed_dict = {tccnn.X: X_train_batch, tccnn.Y: Y_train_batch, tccnn.keep_prob: 0.5}
                sess.run(tccnn.step, feed_dict=feed_dict)
                train_loss += sess.run(tccnn.loss, feed_dict=feed_dict)
                _train_acc = sess.run(tccnn.accuracy, feed_dict=feed_dict)
                if FLAGS.n_tasks == 1:
                    _train_acc = np.array([_train_acc])
                for i in range(FLAGS.n_tasks):
                    train_acc[i] += _train_acc[i]
                        
            test_acc = np.array([0.] * FLAGS.n_tasks)
            test_p = np.array([0.] * FLAGS.n_tasks)
            test_r = np.array([0.] * FLAGS.n_tasks)
            test_fscore = np.array([0.] * FLAGS.n_tasks)
            test_auc = np.array([0.] * FLAGS.n_tasks)
            num_batches_test = X_test.shape[0] // FLAGS.batch_size
            for _n, idx in enumerate(range(num_batches_test)):
                X_test_batch = X_test[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                Y_test_batch = Y_test[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                feed_dict = {tccnn.X: X_test_batch, tccnn.Y: Y_test_batch, tccnn.keep_prob: 1}
                _test_acc = sess.run(tccnn.accuracy, feed_dict=feed_dict)

                _test_p=sess.run(tccnn.p, feed_dict=feed_dict)
                _test_r=sess.run(tccnn.r, feed_dict=feed_dict)
                _test_f1=sess.run(tccnn.f1, feed_dict=feed_dict)
                for i in tccnn.auc_op:
                    sess.run(i, feed_dict=feed_dict)
                _test_auc = sess.run(tccnn.auc, feed_dict=feed_dict)

                if FLAGS.n_tasks == 1:
                    _test_acc = np.array([_test_acc])
                    _test_f1 = np.array([_test_f1])
                    _test_auc = np.array([_test_auc])
                                                
                    _test_p = np.array([_test_p])
                    _test_r = np.array([_test_r])
                        
                if FLAGS.n_tasks == 3:
                    _test_acc = np.array([_test_acc])[0]
                    _test_p = np.array([_test_p])[0]
                    _test_r = np.array([_test_r])[0]
                    _test_auc = np.array([_test_auc])[0]
                for i in range(FLAGS.n_tasks):
                    if FLAGS.n_tasks == 1:
                        test_acc[i] += _test_acc[i]
                        test_fscore[i] += _test_f1[i]
                        test_auc[i] += _test_auc[i]
                                                        
                        test_p[i] += _test_p[i]
                        test_r[i] += _test_r[i]
                            
                    if FLAGS.n_tasks == 3:
                        test_acc[i] += _test_acc[i]
                        test_p[i] += _test_p[i]
                        test_r[i] += _test_r[i]
                        test_auc[i] += _test_auc[i]
            if FLAGS.n_tasks == 1:
                print('Epoch {}, loss={:.6f}, train accuracy = {}, test accuracy '
                      '= {}, test p '
                      '= {}, test r '    
                      '= {}, test fscore '
                      '= {}, test auc'
                      '= {}'.format(0, train_loss/num_batches,
                                        train_acc/num_batches, test_acc/num_batches_test,
                                        test_p/num_batches_test, test_r/num_batches_test,
                                        test_fscore/num_batches_test, test_auc/num_batches_test),file=output_file, flush=True)
            if FLAGS.n_tasks == 3:
                print('Epoch {}, loss={:.6f}, train accuracy = {}, test accuracy '
                      '= {}, test p '
                      '= {}, test r '
                      '= {}, test fscore '
                      '= {}, test auc'
                      '= {}'.format(0, train_loss/num_batches,
                                        train_acc/num_batches, test_acc/num_batches_test,test_p/num_batches_test, test_r/num_batches_test
                                        ,2*test_p*(test_r/num_batches_test)/(test_p+test_r+0.0001), test_auc/num_batches_test),file=output_file, flush=True)
                    
            start_time = time.time()        
            for epoch in range(FLAGS.epoches):
                train_loss, train_acc = .0, np.array([0.] * FLAGS.n_tasks)
                num_batches = X_train.shape[0] // FLAGS.batch_size
                shuffled_batches = choice(range(num_batches), num_batches, replace=False)
                for _n, idx in enumerate(shuffled_batches):
                    X_train_batch = X_train[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                    Y_train_batch = Y_train[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                    feed_dict = {tccnn.X: X_train_batch, tccnn.Y: Y_train_batch, tccnn.keep_prob: 0.5}
                    sess.run(tccnn.step, feed_dict=feed_dict)
                    train_loss += sess.run(tccnn.loss, feed_dict=feed_dict)
                    _train_acc = sess.run(tccnn.accuracy, feed_dict=feed_dict)
                    if FLAGS.n_tasks == 1:
                        _train_acc = np.array([_train_acc])
                    for i in range(FLAGS.n_tasks):
                        train_acc[i] += _train_acc[i]
#                     train_acc += _train_acc
#                     print('Training epoch {:2d}, batch {:4d}/{:4d}, loss = {:.8f}'.format(
#                         epoch+1, _n, num_batches, train_loss/(_n+1)), end='\r', flush=True)

                #if (epoch+1) % 4 != 0:
                #    continue
                

                test_acc = np.array([0.] * FLAGS.n_tasks)
                test_p = np.array([0.] * FLAGS.n_tasks)
                test_r = np.array([0.] * FLAGS.n_tasks)
                test_fscore = np.array([0.] * FLAGS.n_tasks)
                test_auc = np.array([0.] * FLAGS.n_tasks)
                num_batches_test = X_test.shape[0] // FLAGS.batch_size
                for _n, idx in enumerate(range(num_batches_test)):
                    X_test_batch = X_test[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                    Y_test_batch = Y_test[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
#                     print (Y_pred,flush=False)
#                     print (Y_test_batch,flush=False)
                    feed_dict = {tccnn.X: X_test_batch, tccnn.Y: Y_test_batch, tccnn.keep_prob: 1}
                    _test_acc = sess.run(tccnn.accuracy, feed_dict=feed_dict)

                    _test_p=sess.run(tccnn.p, feed_dict=feed_dict)
                    _test_r=sess.run(tccnn.r, feed_dict=feed_dict)
                    _test_f1=sess.run(tccnn.f1, feed_dict=feed_dict)
                    for i in tccnn.auc_op:
                        sess.run(i, feed_dict=feed_dict)
                    _test_auc = sess.run(tccnn.auc, feed_dict=feed_dict)
#                     print (_test_acc,flush=False)
#                     print (_test_f1,flush=False)
                    if FLAGS.n_tasks == 1:
                        _test_acc = np.array([_test_acc])
                        _test_f1 = np.array([_test_f1])
                        _test_auc = np.array([_test_auc])
                                                
                        _test_p = np.array([_test_p])
                        _test_r = np.array([_test_r])
                        
                    if FLAGS.n_tasks == 3:
                        _test_acc = np.array([_test_acc])[0]
                        _test_p = np.array([_test_p])[0]
                        _test_r = np.array([_test_r])[0]
                        _test_auc = np.array([_test_auc])[0]
                    for i in range(FLAGS.n_tasks):
                        if FLAGS.n_tasks == 1:
                            test_acc[i] += _test_acc[i]
                            test_fscore[i] += _test_f1[i]
                            test_auc[i] += _test_auc[i]
                                                        
                            test_p[i] += _test_p[i]
                            test_r[i] += _test_r[i]
                            
                        if FLAGS.n_tasks == 3:
                            test_acc[i] += _test_acc[i]
                            test_p[i] += _test_p[i]
                            test_r[i] += _test_r[i]
                            test_auc[i] += _test_auc[i]
                if FLAGS.n_tasks == 1:
                    print('Epoch {}, loss={:.6f}, train accuracy = {}, test accuracy '
                      '= {}, test p '
                      '= {}, test r '    
                      '= {}, test fscore '
                      '= {}, test auc'
                      '= {}'.format(epoch+1, train_loss/num_batches,
                                        train_acc/num_batches, test_acc/num_batches_test,
                                        test_p/num_batches_test, test_r/num_batches_test,
                                        test_fscore/num_batches_test, test_auc/num_batches_test),file=output_file, flush=True)
                if FLAGS.n_tasks == 3:
                    print('Epoch {}, loss={:.6f}, train accuracy = {}, test accuracy '
                      '= {}, test p '
                      '= {}, test r '
                      '= {}, test fscore '
                      '= {}, test auc'
                      '= {}'.format(epoch+1, train_loss/num_batches,
                                        train_acc/num_batches, test_acc/num_batches_test,test_p/num_batches_test, test_r/num_batches_test
                                        ,2*test_p*(test_r/num_batches_test)/(test_p+test_r+0.0001), test_auc/num_batches_test),file=output_file, flush=True)

                #print('  -> previous accuracy: ' + str(best_accuracy), file=output_file, flush=True)
                #print('  -> latest accuracy: ' + str(test_acc/num_batches_test), file=output_file, flush=True)
                #if sum(test_acc/num_batches_test >= best_accuracy) == FLAGS.n_tasks:
                    #model_fn = 'tccnn-fold_{}'.format(fold_idx)
                    #saver.save(sess, os.path.join(dir_saved_models, model_fn))
                    #print('model {} updated'.format(model_fn), file=output_file, flush=True)
                    #best_accuracy = test_acc / num_batches_test
            
            train_time = time.time() - start_time
            print('training time = {:.6f}'.format(train_time / 60))

if __name__ == '__main__':
    tf.app.run()
