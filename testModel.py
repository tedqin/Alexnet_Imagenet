#!/usr/bin/env python
# coding: UTF-8
from __future__ import division
import os


import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes



path = 'testModel'
withPath = lambda f: '{}/{}'.format(path,f)
testImg = dict((f,cv2.imread(withPath(f))) for f in os.listdir(path) if os.path.isfile(withPath(f)))

id = []
label = []
filename = 'label.txt'
with open(filename, 'r') as fr:
    while True:
        lines = fr.readline()
        if not lines:
            break
            pass
        id_tmp, label_tmp = [i for i in lines.split('\t')]
        id.append(id_tmp)
        label.append(label_tmp)
        pass
    id = np.array(id)
    label = np.array(label)
    pass
label_dict = dict(zip(id, label))

# noinspection PyUnboundLocalVariable

iter_count = 0
num_examples = 3
#num_examples = 1000
if testImg.values():
    #some params
    dropoutPro = 1
    classNum = 1000
    skip = []

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for key,img in testImg.items():
            #img preprocess
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]
            print(key, res + '\n', label_dict[key])
            if res + '\n' == label_dict[key]:
                iter_count += 1
        print("precision:", iter_count / num_examples)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            # print("{}: {}\n----".format(key,res))
            # cv2.imshow("demo", img)
            # cv2.waitKey(0)
