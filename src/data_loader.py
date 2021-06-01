#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# __date__ = 17-10-24:20-08
# __author__ = Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random, logging, os
from multiprocessing import Process, Queue, Manager
import sys
import queue


class DataLoader(object):
    def __init__(self, params, sampler_workers=16):
        self.data_dir = params.data_dir
        self.batch_size = params.batch_size
        self.n_block = params.n_block
        self.max_length = params.max_length
        self.n_cl_neg = params.n_cl_neg
        self.sampler_workers = sampler_workers

        self.block_size = self.max_length // self.n_block

        train_data_path = os.path.join(params.data_dir, 'train_data.csv')
        test_csv_path = os.path.join(params.data_dir, 'test_data.csv')

        self.preload_feat_into_memory()
        if params.phase == 'train':
            self.read_train_data(train_data_path)
            self.epoch_train_data = self.generate_train_data()
            self.initTrainProcess()
        self.read_test_data(test_csv_path)
        self.test_queue = Queue(maxsize=self.sampler_workers * 10)
        if params.phase == 'test':
            self.initTestProcess()

    def initTrainProcess(self):
        self.train_processors = []
        self.n_train_batch = 0
        self.train_queue = Queue(maxsize=self.sampler_workers * 10)
        random.shuffle(self.epoch_train_data)
        samples_per_worker = self.epoch_train_length // self.sampler_workers

        for i in range(self.sampler_workers):
            if i == self.sampler_workers - 1:
                first = i * samples_per_worker
                last = self.epoch_train_length
            else:
                first = i * samples_per_worker
                last = (i + 1) * samples_per_worker

            if last >= first:
                self.n_train_batch += (last - first + 1) // self.batch_size

                self.train_processors.append(Process(target=self.processTrainBatch, args=(first, last)))
                self.train_processors[-1].daemon = True
                self.train_processors[-1].start()

    def processTrainBatch(self, first, last):
        data = self.epoch_train_data[first:last]

        while True:
            random.shuffle(data)
            n_batch = len(data) // self.batch_size
            for i in range(n_batch):
                batch_data = data[i * self.batch_size: (i + 1) * self.batch_size]
                user_ids, item_ids, cate_ids, labels = zip(*batch_data)
                att_iids, att_cids, intra_mask, inter_mask = self.get_att_ids(user_ids)
                cl_negs = self.get_neg_ids(user_ids)
                self.train_queue.put(
                    (user_ids, item_ids, cate_ids, att_iids, att_cids, intra_mask, inter_mask, labels, cl_negs)
                )

    def get_train_batch(self):
        return self.train_queue.get()

    def initTestProcess(self):
        self.test_processors = []
        self.n_test_batch = 0
        samples_per_worker = self.epoch_test_length // self.sampler_workers

        for i in range(self.sampler_workers):
            if i == self.sampler_workers - 1:
                first = i * samples_per_worker
                last = self.epoch_test_length
            else:
                first = i * samples_per_worker
                last = (i + 1) * samples_per_worker

            if last >= first:
                self.n_test_batch += (last - first + 1) // self.batch_size
                self.test_processors.append(Process(target=self.processTestBatch, args=(first, last)))
                self.test_processors[-1].daemon = True
                self.test_processors[-1].start()

    def processTestBatch(self, first, last):
        data = self.test_data[first:last]
        n_batch = len(data) // self.batch_size
        for i in range(n_batch):
            batch_data = data[i * self.batch_size: (i + 1) * self.batch_size]
            user_ids, item_ids, cate_ids, labels = zip(*batch_data)
            item_vecs = self.get_test_cover_img_feature(item_ids)
            att_iids, att_cids, intra_mask, inter_mask = self.get_att_ids(user_ids)
            cl_negs = self.get_neg_ids(user_ids)
            self.test_queue.put(
                (user_ids, item_vecs, cate_ids, att_iids, att_cids, intra_mask, inter_mask, labels, item_ids, cl_negs)
            )

    def get_test_batch(self):
        return self.test_queue.get()

    def close_train_processes(self):
        try:
            for p in self.train_processors:
                p.terminate()
                p.join()
            del self.train_queue
        except:
            raise ValueError('Error when close train processes')

    def close_test_processes(self):
        try:
            for p in self.test_processors:
                p.terminate()
                p.join()
        except:
            raise ValueError('Error when close test processes')

    def generate_train_data(self, neg_ratio=3):
        logging.info('generate samples for training')
        epoch_train_data = []
        for item in self.train_data:
            pos_num, neg_num = len(item[0]), len(item[1])
            epoch_train_data.extend(item[0])
            if neg_num < pos_num * neg_ratio:
                epoch_train_data.extend(item[1])
            else:
                epoch_train_data.extend(random.sample(item[1], int(pos_num * neg_ratio)))
        random.shuffle(epoch_train_data)
        self.epoch_train_length = len(epoch_train_data)
        return epoch_train_data

    def read_train_data(self, data_list_path):
        logging.info('start read data list from disk')
        with open(data_list_path, 'r') as reader:
            reader.readline()
            raw_data = map(lambda x: x.strip('\n').split(','), reader.readlines())

        data = [[[], []] for _ in range(10986)]
        for item in raw_data:
            if int(item[3]) == 1:
                data[int(item[0])][0].append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
            else:
                data[int(item[0])][1].append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
        self.train_data = data

    def read_test_data(self, test_data_path, sep=','):
        with open(test_data_path, 'r') as reader:
            reader.readline()
            lines = map(lambda x: x.strip('\n').split(sep), reader.readlines())
            data = map(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3])), lines)

        self.test_data = list(data)
        self.epoch_test_length = len(self.test_data)
        logging.info('{} test samples'.format(self.epoch_test_length))

    def get_neg_ids(self, user_ids):
        '''
        sample neg samples for cl loss
        Args:
            user_ids:

        Returns:

        '''
        result = []
        for uid in user_ids:
            neg = self.true_negs[uid]
            # result.append(list(random.sample(list(neg)*10+list(self.neg_buffers[uid]), self.n_cl_neg)))

            if len(neg) >= self.n_cl_neg:
                result.append(list(random.sample(neg, self.n_cl_neg)))
            else:
                result.append(list(neg) + list(random.sample(self.neg_buffers[uid], self.n_cl_neg - len(neg))))
        return result

    def sample_vid(self, tuples):
        item_ids, cate_ids, timestamps = list(zip(*tuples))
        length = len(item_ids)
        padding_num = self.max_length - length
        if padding_num > 0:
            item_ids = list(item_ids) + [984983] * padding_num
            cate_ids = list(cate_ids) + [512] * padding_num
            intra_mask = [1] * length + [0] * padding_num
            pad_n_block = padding_num // self.block_size
            inter_mask = [1] * (self.n_block - pad_n_block) + [0] * pad_n_block
        else:
            indices = random.sample(list(range(length)), self.max_length)
            indices.sort()
            item_ids = [item_ids[i] for i in indices]
            cate_ids = [cate_ids[i] for i in indices]
            intra_mask = [1] * self.max_length
            inter_mask = [1] * self.n_block

        return item_ids, cate_ids, intra_mask, inter_mask

    def get_att_ids(self, user_ids):
        xx = [self.sample_vid(self.user_click_ids[uid]) for idx, uid in enumerate(user_ids)]
        batch_att_iids, batch_att_cids, batch_intra_mask, batch_inter_mask = zip(*xx)
        return batch_att_iids, batch_att_cids, batch_intra_mask, batch_inter_mask

    def get_test_cover_img_feature(self, vids):
        head_vec = [self.test_visual_feature[i] for i in vids]
        return head_vec

    def gen_true_negs(self):
        true_negs = {}
        for user_id in range(len(self.user_unclick_ids)):
            item_ids, cate_ids, timestamps = list(zip(*self.user_unclick_ids[user_id]))
            true_negs[user_id] = item_ids
        return true_negs

    def sample_neg_worker(self, user_ids, neg_buffers):
        for user_id in user_ids:
            pos_items = list(map(lambda x: x[0], self.user_click_ids[user_id]))
            candidates = set(range(984983)) - set(self.true_negs[user_id]) - set(pos_items)
            neg_buffers[user_id] = random.sample(candidates, 10000)

    def pre_sample_negs(self):
        print('enter pre_sample_negs function')
        neg_buffers = Manager().dict()
        users = list(self.true_negs.keys())
        #users = [uid for uid in self.true_negs if len(self.true_negs[uid]) < self.n_cl_neg]
        print(len(users))
        n_workers = min(30, self.sampler_workers * 4)
        users_per_worker = len(users) // n_workers
        processors = []
        for i in range(n_workers):
            if i == n_workers - 1:
                tar_users = users[i * users_per_worker:]
                processors.append(Process(target=self.sample_neg_worker, args=(tar_users, neg_buffers)))
            else:
                tar_users = users[i * users_per_worker:(i + 1) * users_per_worker]
                processors.append(Process(target=self.sample_neg_worker, args=(tar_users, neg_buffers)))
            processors[-1].daemon = True
            processors[-1].start()

        for proc in processors:
            proc.join()
        return neg_buffers

    def preload_feat_into_memory(self):
        train_feature_path = os.path.join(self.data_dir, 'train_cover_image_feature.npy')
        test_feature_path = os.path.join(self.data_dir, 'test_cover_image_feature.npy')

        logging.info('load train visual feature')
        train_visual_feature = np.load(train_feature_path)
        self.train_visual_feature = np.concatenate([train_visual_feature, [[0.0] * 512]], axis=0)

        logging.info('load test visual feature')
        self.test_visual_feature = np.load(test_feature_path)

        user_click_ids_path = os.path.join(self.data_dir, 'user_click_ids.npy')
        self.user_click_ids = np.load(user_click_ids_path, allow_pickle=True)

        user_unclick_ids_path = os.path.join(self.data_dir, 'user_unclick_ids.npy')
        self.user_unclick_ids = np.load(user_unclick_ids_path, allow_pickle=True)

        self.true_negs = self.gen_true_negs()
        self.neg_buffers = self.pre_sample_negs()

    def del_temp(self):
        del self.train_visual_feature
