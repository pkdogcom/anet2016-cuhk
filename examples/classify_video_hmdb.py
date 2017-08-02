#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This scripts demos how to do single video classification using the framework
Before using this scripts, please download the model files using

bash models/get_reference_models.sh

Usage:

python classify_video.py <video name>
"""

import os
anet_home = os.environ['ANET_HOME']
import sys
sys.path.append(anet_home)
import io

from pyActionRec.action_classifier import ActionClassifier
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("video_name", type=str)
parser.add_argument("--use_flow", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

VIDEO_NAME = args.video_name
USE_FLOW = args.use_flow
GPU=args.gpu

models=[]

models = [('models/hmdb51/tsn_bn_inception_rgb_deploy.prototxt',
           'models/hmdb51_split_1_tsn_rgb_reference_bn_inception.caffemodel',
           1.0, 0, False, 224)]


if USE_FLOW:
    models.append(('models/hmdb51/tsn_bn_inception_flow_deploy.prototxt',
                   'models/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel',
                   0.2, 1, False, 224))

cls = ActionClassifier(models, dev_id=GPU)
rst = cls.classify(VIDEO_NAME)

scores = rst[0]

with io.open(os.path.join(anet_home,"data/hmdb51_splits/class_list.txt"), encoding='utf8') as f:
    lb_list = f.read().splitlines()

with io.open(os.path.join(anet_home,"data/hmdb51_splits/class_list_cn.txt"), encoding='utf8') as f:
    list_cn = f.read().splitlines()

idx = np.argsort(scores)[::-1]

print '----------------Classification Results----------------------'
for i in xrange(10):
    k = idx[i]
    print "{}  {} : {}".format(lb_list[k], list_cn[k].encode('utf-8'), scores[k])


