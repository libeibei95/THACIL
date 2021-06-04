#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from nets import var_init, dense
from nets import temporal_hierarchical_attention
from nets import dnn, vanilla_attention


class Model(object):
    def __init__(self, param):
        self.batch_size = param.batch_size
        self.reg = param.reg
        self.dropout = param.dropout
        self.batch_size = param.batch_size
        self.max_length = param.max_length
        self.num_heads = param.num_heads
        self.n_block = param.n_block
        self.n_cl_neg = param.n_cl_neg

        self.fusion_layers = param.fusion_layers
        self.optimizer = param.optimizer

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.user_dim = param.user_dim
        self.item_dim = param.item_dim
        self.cate_dim = param.cate_dim
        self.max_gradient_norm = param.max_gradient_norm

        self.init_embedding()
        self.set_placeholder()
        self.set_optimizer()
        with tf.device('/gpu:0'):
            with tf.variable_scope('THACIL'):
                self.train_inference()
                tf.get_variable_scope().reuse_variables()
                self.test_inference()

    def train_inference(self):
        item_vec = self.get_train_cover_image_feature(self.item_ids_ph)
        loss, acc, _ = self.build_model(item_vec,
                                        self.cate_ids_ph,
                                        self.att_iids_ph,
                                        self.att_cids_ph,
                                        self.intra_mask_ph,
                                        self.inter_mask_ph,
                                        self.labels_ph,
                                        self.cl_pos_ph,
                                        self.cl_mask_ph,
                                        self.cl_neg_ph,
                                        self.dropout)
        # train op
        train_params = tf.trainable_variables()
        gradients = tf.gradients(loss, train_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, train_params), self.global_step)
        self.train_loss = loss
        self.train_acc = acc

        # summary
        self.train_summuries = tf.summary.merge([tf.summary.scalar('train/loss', loss),
                                                 tf.summary.scalar('train/acc', acc),
                                                 tf.summary.scalar('lr', self.lr_ph)])

        # saver
        params = [v for v in tf.trainable_variables() if 'adam' not in v.name]
        self.saver = tf.train.Saver(params, max_to_keep=1)

    def test_inference(self):
        loss, acc, logits = self.build_model(self.item_vec_ph,
                                             self.cate_ids_ph,
                                             self.att_iids_ph,
                                             self.att_cids_ph,
                                             self.intra_mask_ph,
                                             self.inter_mask_ph,
                                             self.labels_ph,
                                             self.cl_pos_ph,
                                             self.cl_mask_ph,
                                             self.cl_neg_ph,
                                             1.0)

        self.test_loss = loss
        self.test_acc = acc
        self.test_logits = logits

        # summary
        self.test_summuries = tf.summary.merge([tf.summary.scalar('test/acc', acc),
                                                tf.summary.scalar('test/loss', loss)])

    # def seq_cl_loss(self,
    #                 pos_item_emb,
    #                 neg_item_emb,
    #                 intra_mask,
    #                 t=1):
    #     '''
    #
    #     Args:
    #       pos_item_emb: batch_size * seq_len * embed_size
    #       neg_item_emb: batch_size * n_cl_neg * embed_size
    #       intra_mask: batch_size * seq_len
    #       t: scalar
    #     Returns:
    #     '''
    #
    #     pos_item_emb = tf.nn.l2_normalize(pos_item_emb, -1)
    #     neg_item_emb = tf.nn.l2_normalize(neg_item_emb, -1)
    #
    #     pos_item_emb2 = tf.concat([pos_item_emb[:, 1:, :], pos_item_emb[:, -2:-1, :]], axis=1)
    #     # batch_size * seq_len
    #     pos_scores = tf.exp(tf.reduce_sum(tf.multiply(pos_item_emb, pos_item_emb2), axis=-1) / t)
    #     # batch_size * seq_len * n_cl_neg
    #     neg_scores = tf.exp(tf.matmul(pos_item_emb, tf.transpose(neg_item_emb, [0, 2, 1])) / t)
    #     all_scores = tf.concat([tf.expand_dims(pos_scores, axis=-1), neg_scores], axis=-1)
    #     all_scores = tf.reduce_sum(all_scores, axis=-1)
    #     ssl_loss = - tf.log(pos_scores / all_scores)  # batch_size * seq_len
    #
    #     avg_mask = tf.reduce_sum(tf.cast(intra_mask, dtype=tf.float32), axis=1)
    #     ssl_loss = tf.reduce_sum(tf.multiply(ssl_loss, tf.cast(intra_mask, dtype=tf.float32)), axis=1)
    #     ssl_loss = tf.reduce_mean(ssl_loss / avg_mask)
    #     return ssl_loss

    """
    def seq_cl_loss(self,
                    tar_item_vec,
                    pos_item_vec,
                    neg_item_vec,
                    t=1):
        '''
        
        Args:
            tar_item_vec: batch_size * embed_size
            pos_item_vec: batch_size * embed_size
            neg_item_vec: batch_size * n_neg * embed_size
            t: 

        Returns:

        '''                    
            
        all_item_vec = tf.concat([tf.expand_dims(pos_item_vec, axis=1), neg_item_vec], axis=1)
        tar_item_vec = tf.nn.l2_normalize(tar_item_vec, -1)
        all_item_vec = tf.nn.l2_normalize(all_item_vec, -1)
        all_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(tar_item_vec, axis=1), all_item_vec), axis=-1)
        pos_scores = tf.exp(all_scores[:, 0, :] / t)
        all_scores = tf.reduce_sum(tf.exp(all_scores / t), axis=-1)
        ssl_loss = - tf.reduce_mean(tf.log(pos_scores / all_scores))
        return ssl_loss
    """

    """
    def seq_cl_loss(self,
                    tar_item_vec,
                    pos_item_vec,
                    neg_item_vec,
                    t=1):
        '''

        Args:
            tar_item_vec: batch_size * embed_size
            pos_item_vec: batch_size * embed_size
            neg_item_vec: n_neg * embed_size
            t:

        Returns:

        '''
        tar_item_vec = tf.nn.l2_normalize(tar_item_vec, -1)
        pos_item_vec = tf.nn.l2_normalize(pos_item_vec, -1)
        neg_item_vec = tf.nn.l2_normalize(neg_item_vec, -1)
        # (batch_size, )
        pos_scores = tf.exp(tf.reduce_sum(tf.multiply(tar_item_vec, pos_item_vec), axis=-1) / t)
        # (batch_size, n_neg)
        neg_scores = tf.squeeze(tf.matmul(tar_item_vec, tf.transpose(neg_item_vec, [1, 0])))
        # (batch_size, )
        neg_scores = tf.reduce_sum(tf.exp(neg_scores / t), axis=-1)
        # (1, )
        ssl_loss = -tf.reduce_mean(tf.log(pos_scores / (pos_scores + neg_scores)))
        return ssl_loss
    """

    def seq_cl_loss(self,
                    base_item_vec,
                    pos_item_vec,
                    pos_mask,
                    neg_item_vec,
                    t=1):
        '''

        Args:
            base_item_vec: batch_size * seq_len * embed_size
            pos_item_vec: batch_size * seq_len * embed_size
            pos_mask: batch_size * seq_len
            neg_item_vec: n_cl_negs * embed_size

        Returns:

        '''

        base_item_vec = tf.nn.l2_normalize(base_item_vec, -1)
        pos_item_vec = tf.nn.l2_normalize(pos_item_vec, -1)
        neg_item_vec = tf.nn.l2_normalize(neg_item_vec, -1)

        batch_size, seq_len, emb_size = base_item_vec.shape

        # shape(batch_size, seq_len)
        pos_scores = tf.exp(tf.reduce_sum(tf.multiply(base_item_vec, pos_item_vec), axis=-1) / t)
        # shape(batch_size, seq_len, n_cl_negs)
        neg_scores = tf.multiply(tf.reshape(base_item_vec, [-1, emb_size]),
                                 tf.transpose(neg_item_vec, [1, 0]))
        # shape(batch_size, seq_len)
        neg_scores = tf.reduce_sum(tf.exp(tf.reshape(neg_scores, shape=[batch_size, seq_len]) / t), axis=1)
        pos_mask = tf.cast(pos_mask, dtype=tf.float32)  # batch_size * seq_len
        avg_mask = tf.reduce_sum(pos_mask, axis=1)  # batch_size * seq_len
        ssl_loss = tf.reduce_sum(tf.multiply(- tf.log(pos_scores / (pos_scores + neg_scores)), pos_mask),
                                 axis=-1) / avg_mask
        ssl_loss = tf.reduce_mean(ssl_loss)
        return ssl_loss

    def build_model(self,
                    item_vec,
                    cate_ids,
                    att_iids,
                    att_cids,
                    intra_mask,
                    inter_mask,
                    labels,
                    cl_pos_iids,
                    cl_mask,
                    cl_neg_iids,
                    keep_prob):

        with tf.variable_scope('item_embedding'):
            att_item_vec = self.get_train_cover_image_feature(att_iids)
            att_cate_emb = self.get_cate_emb(att_cids)

            pos_item_vec = self.get_train_cover_image_feature(cl_pos_iids)
            neg_item_vec = self.get_train_cover_image_feature(cl_neg_iids)

            item_vec = dense(item_vec, self.item_dim, ['w1'], 1.0)
            att_item_vec = dense(att_item_vec, self.item_dim, ['w1'], 1.0, reuse=True)
            pos_item_vec = dense(pos_item_vec, self.item_dim, ['w1'], 1.0, reuse=True)
            neg_item_vec = dense(neg_item_vec, self.item_dim, ['w1'], 1.0, reuse=True)

            cate_emb = self.get_cate_emb(cate_ids)
            item_emb = tf.concat([item_vec, cate_emb], axis=-1)

        with tf.variable_scope('cl_loss'):
            # att_item_emb = tf.concat([att_item_vec, att_cate_emb], axis=-1)
            # neg_item_emb = tf.concat([neg_item_vec, neg_cate_emb], axis=-1)
            seq_cl_loss = self.seq_cl_loss(att_item_vec, pos_item_vec, cl_mask, neg_item_vec)

        with tf.variable_scope('temporal_hierarchical_attention'):
            user_profiles = temporal_hierarchical_attention(att_cate_emb,
                                                            att_item_vec,
                                                            intra_mask,
                                                            inter_mask,
                                                            self.num_heads,
                                                            keep_prob)

        with tf.variable_scope('micro_video_click_through_prediction'):
            user_profile = vanilla_attention(user_profiles, item_emb, inter_mask, keep_prob)
            y = dnn(tf.concat([user_profile, item_emb], axis=-1), self.fusion_layers, keep_prob)
        logits = y

        # regularization
        emb_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'embedding' in v.name])
        w_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name])
        l2_norm = emb_l2_loss + w_l2_loss

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels)
        ) + l2_norm * self.reg + 1 * seq_cl_loss
        acc = self.compute_acc(logits, self.labels_ph)
        return loss, acc, logits

    def train(self, sess, data, lr):
        feed_dicts = {
            self.user_ids_ph: data[0],
            self.item_ids_ph: data[1],
            self.cate_ids_ph: data[2],
            self.att_iids_ph: data[3],
            self.att_cids_ph: data[4],
            self.intra_mask_ph: data[5],
            self.inter_mask_ph: data[6],
            self.labels_ph: data[7],
            self.lr_ph: lr,
            self.cl_pos_ph: data[8],
            self.cl_mask_ph: data[9],
            self.cl_neg_ph: data[10]
        }
        train_run_op = [self.train_loss, self.train_acc, self.train_summuries, self.train_op]
        loss, acc, summaries, _ = sess.run(train_run_op, feed_dicts)
        return loss, acc, summaries

    def test(self, sess, data):
        feed_dicts = {
            self.user_ids_ph: data[0],
            self.item_ids_ph: data[0],  # ignore item_ids_ph in test phase
            self.item_vec_ph: data[1],
            self.cate_ids_ph: data[2],
            self.att_iids_ph: data[3],
            self.att_cids_ph: data[4],
            self.intra_mask_ph: data[5],
            self.inter_mask_ph: data[6],
            self.cl_pos_ph: data[7],
            self.cl_mask_ph: data[8],
            self.cl_neg_ph: data[9],
            self.labels_ph: data[10]
        }
        test_run_op = [self.test_loss, self.test_logits, self.test_acc, self.test_summuries]
        loss, logits, acc, summaries = sess.run(test_run_op, feed_dicts)
        return loss, logits, acc, summaries

    def save(self, sess, model_path, epoch):
        self.saver.save(sess, model_path, global_step=epoch)
        logging.info("Saved model in epoch {}".format(epoch))

    def compute_acc(self, logit, labels):
        pred = tf.cast(tf.nn.sigmoid(logit) >= 0.5, tf.float32)
        correct_pred = tf.equal(pred, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def set_placeholder(self):
        self.user_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.item_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.cate_ids_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.att_iids_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.att_cids_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.intra_mask_ph = tf.placeholder(tf.bool, shape=(self.batch_size, self.max_length))
        self.inter_mask_ph = tf.placeholder(tf.bool, shape=(self.batch_size, self.n_block))
        self.item_vec_ph = tf.placeholder(tf.float32, shape=(self.batch_size, 512))
        self.labels_ph = tf.placeholder(tf.float32, shape=(self.batch_size))
        self.lr_ph = tf.placeholder(tf.float32, shape=())
        self.cl_pos_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.cl_mask_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.cl_neg_ph = tf.placeholder(tf.int32, shape=(self.n_cl_neg,))

    def init_embedding(self):
        category_embedding = var_init('category_embedding', [512, self.cate_dim], tf.random_normal_initializer())
        self.category_embedding = tf.concat([category_embedding, tf.zeros((1, self.cate_dim))], axis=0)
        self.train_visual_emb = tf.Variable(tf.constant(0.0, shape=[984984, 512]), trainable=False,
                                            name='train_visual_emb')

    def restore_train_visual_emb(self, visual_feature, sess):
        visual_ph = tf.placeholder(tf.float32, [984984, 512])
        emb_init = self.train_visual_emb.assign(visual_ph)
        sess.run(emb_init, feed_dict={visual_ph: visual_feature})
        logging.info('load train visual feature into GPU memory successfully')

    def get_cate_emb(self, cate_ids):
        return tf.nn.embedding_lookup(self.category_embedding, cate_ids)

    def get_train_cover_image_feature(self, item_ids):
        return tf.nn.embedding_lookup(self.train_visual_emb, item_ids)

    def set_optimizer(self):
        if self.optimizer == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr_ph)
        elif self.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
        else:
            raise ValueError('do not support {} optimizer'.format(self.optimizer))
