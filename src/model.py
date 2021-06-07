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
        self.neg_flag = param.neg_flag

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
        full_loss, _, acc, _ = self.build_model(item_vec,
                                                self.cate_ids_ph,
                                                self.att_iids_ph,
                                                self.att_cids_ph,
                                                self.intra_mask_ph,
                                                self.inter_mask_ph,
                                                self.labels_ph,
                                                self.dropout,
                                                self.att_iids_ph2,
                                                self.att_cids_ph2,
                                                self.intra_mask_ph2,
                                                self.inter_mask_ph2,
                                                self.neg_att_iids_ph,
                                                self.neg_att_cids_ph,
                                                self.neg_intra_mask_ph,
                                                self.neg_inter_mask_ph)
        # train op
        train_params = tf.trainable_variables()
        gradients = tf.gradients(full_loss, train_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, train_params), self.global_step)
        self.train_loss = full_loss
        self.train_acc = acc

        # summary
        self.train_summuries = tf.summary.merge([tf.summary.scalar('train/loss', full_loss),
                                                 tf.summary.scalar('train/acc', acc),
                                                 tf.summary.scalar('lr', self.lr_ph)])

        # saver
        params = [v for v in tf.trainable_variables() if 'adam' not in v.name]
        self.saver = tf.train.Saver(params, max_to_keep=1)

    def test_inference(self):
        _, loss, acc, logits = self.build_model(self.item_vec_ph,
                                                self.cate_ids_ph,
                                                self.att_iids_ph,
                                                self.att_cids_ph,
                                                self.intra_mask_ph,
                                                self.inter_mask_ph,
                                                self.labels_ph,
                                                1.0,
                                                self.att_iids_ph2,
                                                self.att_cids_ph2,
                                                self.intra_mask_ph2,
                                                self.inter_mask_ph2,
                                                self.neg_att_iids_ph,
                                                self.neg_att_cids_ph,
                                                self.neg_intra_mask_ph,
                                                self.neg_inter_mask_ph)

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

    def user_cl_loss(self, user_emb1, user_emb2, t=1):
        '''
        Calculating SSL loss
        '''
        # batch_users, _ = tf.unique(self.users)
        print(user_emb1.shape, user_emb2.shape)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, axis=-1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, axis=-1)
        normalize_user_emb2_neg = normalize_user_emb2

        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=-1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_user_emb2_neg, transpose_a=False, transpose_b=True)

        pos_score_user = tf.exp(pos_score_user / t)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / t), axis=-1)

        ssl_loss = -tf.reduce_mean(tf.log(pos_score_user / ttl_score_user))
        # ssl_loss = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))
        return ssl_loss

    def triple_cl_loss(self, user_emb1, user_emb2, neg_emb, t=1):
        user_emb1 = tf.nn.l2_normalize(user_emb1, axis=-1)
        user_emb2 = tf.nn.l2_normalize(user_emb2, axis=-1)
        neg_emb = tf.nn.l2_normalize(neg_emb, axis=-1)
        # pos_sim_user = tf.reduce_sum(tf.multiply(user_emb1, user_emb2), axis=-1)
        # neg_sim_user = tf.reduce_sum(tf.multiply(user_emb1, neg_emb), axis=-1)
        pos_dis_user = 1 - tf.reduce_sum(tf.multiply(user_emb1, user_emb2), axis=-1)
        neg_dis_user = 1 - tf.reduce_sum(tf.multiply(user_emb1, neg_emb), axis=-1)

        # ssl_loss = tf.reduce_mean(-tf.maximum(neg_sim_user + 1 - pos_sim_user, 0))
        ssl_loss = tf.reduce_mean(tf.maximum(tf.square(pos_dis_user) + 2 - tf.square(neg_dis_user), 0))
        # ssl_loss = tf.reduce_mean(tf.square(pos_dis_user)) + tf.reduce_mean(
        #     tf.square(tf.maximum((1 - neg_dis_user), 0)))
        return ssl_loss

    def build_model(self,
                    item_vec,
                    cate_ids,
                    att_iids,
                    att_cids,
                    intra_mask,
                    inter_mask,
                    labels,
                    keep_prob,
                    att_iids2,
                    att_cids2,
                    intra_mask2,
                    inter_mask2,
                    neg_att_iids,
                    neg_att_cids,
                    neg_intra_mask,
                    neg_inter_mask):

        with tf.variable_scope('item_embedding'):
            att_item_vec = self.get_train_cover_image_feature(att_iids)
            att_cate_emb = self.get_cate_emb(att_cids)
            att_item_vec2 = self.get_train_cover_image_feature(att_iids2)
            att_cate_emb2 = self.get_cate_emb(att_cids2)
            neg_item_vec = self.get_train_cover_image_feature(neg_att_iids)
            neg_cate_emb = self.get_cate_emb(neg_att_cids)

            item_vec = dense(item_vec, self.item_dim, ['w1'], 1.0)
            att_item_vec = dense(att_item_vec, self.item_dim, ['w1'], 1.0, reuse=True)
            att_item_vec2 = dense(att_item_vec2, self.item_dim, ['w1'], 1.0, reuse=True)
            neg_item_vec = dense(neg_item_vec, self.item_dim, ['w1'], 1.0, reuse=True)

            cate_emb = self.get_cate_emb(cate_ids)
            item_emb = tf.concat([item_vec, cate_emb], axis=-1)

        with tf.variable_scope('temporal_hierarchical_attention', reuse=tf.AUTO_REUSE):
            user_profiles = temporal_hierarchical_attention(att_cate_emb,
                                                            att_item_vec,
                                                            intra_mask,
                                                            inter_mask,
                                                            self.num_heads,
                                                            keep_prob)

            user_profiles2 = temporal_hierarchical_attention(att_cate_emb2,
                                                             att_item_vec2,
                                                             intra_mask2,
                                                             inter_mask2,
                                                             self.num_heads,
                                                             keep_prob)
            neg_user_profiles = temporal_hierarchical_attention(neg_cate_emb,
                                                                neg_item_vec,
                                                                neg_intra_mask,
                                                                neg_inter_mask,
                                                                self.num_heads,
                                                                keep_prob)

        with tf.variable_scope('micro_video_click_through_prediction', reuse=tf.AUTO_REUSE):
            user_profile = vanilla_attention(user_profiles, item_emb, inter_mask, keep_prob)
            user_profile2 = vanilla_attention(user_profiles2, item_emb, inter_mask2, keep_prob)
            neg_user_profile = vanilla_attention(neg_user_profiles, item_emb, neg_inter_mask, keep_prob)

            if self.neg_flag == 1:
                user_cl_loss = self.user_cl_loss(user_profile, user_profile2)
            elif self.neg_flag == 2:
                user_cl_loss = self.triple_cl_loss(user_profile, user_profile2, neg_user_profile)
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
        ) + l2_norm * self.reg

        full_loss = loss + 1 * user_cl_loss
        acc = self.compute_acc(logits, self.labels_ph)
        return full_loss, loss, acc, logits

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
            # self.cl_neg_ph: data[8],
            self.att_iids_ph2: data[8],
            self.att_cids_ph2: data[9],
            self.intra_mask_ph2: data[10],
            self.inter_mask_ph2: data[11],
            self.neg_att_iids_ph: data[12],
            self.neg_att_cids_ph: data[13],
            self.neg_intra_mask_ph: data[14],
            self.neg_inter_mask_ph: data[15]
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
            self.att_iids_ph2: data[7],
            self.att_cids_ph2: data[8],
            self.intra_mask_ph2: data[9],
            self.inter_mask_ph2: data[10],
            self.neg_att_iids_ph: data[11],
            self.neg_att_cids_ph: data[12],
            self.neg_intra_mask_ph: data[13],
            self.neg_inter_mask_ph: data[14],
            self.labels_ph: data[15],
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
        self.cl_neg_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.n_cl_neg))
        self.att_iids_ph2 = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.att_cids_ph2 = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.intra_mask_ph2 = tf.placeholder(tf.bool, shape=(self.batch_size, self.max_length))
        self.inter_mask_ph2 = tf.placeholder(tf.bool, shape=(self.batch_size, self.n_block))
        self.neg_att_iids_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.neg_att_cids_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_length))
        self.neg_intra_mask_ph = tf.placeholder(tf.bool, shape=(self.batch_size, self.max_length))
        self.neg_inter_mask_ph = tf.placeholder(tf.bool, shape=(self.batch_size, self.n_block))

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
