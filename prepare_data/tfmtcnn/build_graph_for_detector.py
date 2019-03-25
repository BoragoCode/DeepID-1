# coding:utf-8
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from detectors import Detector, FcnDetector, MtcnnDetector
from models import P_Net, R_Net, O_Net

prefix_src = ['./models/mtcnn/modelfile/PNet/PNet', './models/mtcnn/modelfile/RNet/RNet', './models/mtcnn/modelfile/ONet/ONet']
prefix_dst = ["./models/mtcnn/build_graph/PNet/PNet", "./models/mtcnn/build_graph/RNet/RNet", "./models/mtcnn/build_graph/ONet/ONet",]
epoch = [18, 14, 16]
model_src = ['%s-%s' % (x, y) for x, y in zip(prefix_src, epoch)]
model_dst = ['%s-%s' % (x, y) for x, y in zip(prefix_dst, epoch)]

# PNet = FcnDetector(P_Net, model_src[0]);       detectors[0] = PNet
# RNet = Detector(R_Net, 24, 1, model_src[1]);   detectors[1] = RNet
# ONet = Detector(O_Net, 48, 1, model_src[2]);   detectors[2] = ONet

def build_pnet_graph(net_factory, model_src, model_dst):
    graph = tf.Graph()
    with graph.as_default():

        image_op = tf.placeholder(tf.float32, name='input_image')
        width_op = tf.placeholder(tf.int32, name='image_width')
        height_op = tf.placeholder(tf.int32, name='image_height')
        image_reshape = tf.reshape(image_op, [1, height_op, width_op, 3])
        cls_prob, bbox_pred, _ = net_factory(image_reshape, training=False)
            
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            
        saver = tf.train.Saver()
        # ----- check -----
        model_dict = '/'.join(model_src.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(model_dict)
        print(model_src)
        readstate = ckpt and ckpt.model_checkpoint_path
        assert  readstate, "the params dictionary is not valid"
        print("restore models' param")
        # -----------------
        saver.restore(sess, model_src)

        saver.save(sess, model_dst)

def build_rnet_onet_graph(net_factory, data_size, batch_size, model_src, model_dst):
    graph = tf.Graph()
    with graph.as_default():

        image_op = tf.placeholder(tf.float32, 
                                shape=[batch_size, data_size, data_size, 3], 
                                name='input_image')
        cls_prob, bbox_pred, landmark_pred = net_factory(image_op, training=False)
            

        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            

        saver = tf.train.Saver()
        # ----- check -----
        model_dict = '/'.join(model_src.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(model_dict)
        print(model_src)
        readstate = ckpt and ckpt.model_checkpoint_path
        assert  readstate, "the params dictionary is not valid"
        print("restore models' param")
        # -----------------
        saver.restore(sess, model_src)

        saver.save(sess, model_dst)

if __name__ == "__main__":
    build_pnet_graph(P_Net, model_src[0], model_dst[0])
    build_rnet_onet_graph(R_Net, 24, 1, model_src[1], model_dst[1])
    build_rnet_onet_graph(O_Net, 48, 1, model_src[2], model_dst[2])
