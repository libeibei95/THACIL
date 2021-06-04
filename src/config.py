#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config:
    PHASE = 'train'

    BATCH_SIZE =256
    MAX_EPOCH = 40
    INITIAL_LEARNING_RATE = 0.001
    REG = 0.00005
    DROPOUT = 0.8

    DATA_DIR = '../data/input'
    OUT_DIR = '../data/output2'
    RESTORE_MODEL_DIR = '../data/input'
    OPTIMIZER = 'adam'

    NEG_RATIO = 1.0
    FUSION_LAYERS = [128]

    MAX_LENGTH = 160
    ITEM_DIM = 64
    CATE_DIM = 64
    USER_DIM = 128
    DISPLAY = 1000
    NUM_HEADS = 8
    N_BLOCK = 8
    N_CL_NEG = 2048
