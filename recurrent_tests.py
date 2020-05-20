#!/usr/bin/env python3

import os
import unittest
import numpy as np
import tensorflow as tf
import _recurrent_grad
import tensorflow.contrib.slim as slim
recurrent_module = tf.load_op_library('build/librecurrent.so')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RecurrentOpTest(unittest.TestCase):
    
    def test_recurrentGrad_cpu(self):

        config = tf.ConfigProto(log_device_placement = True)
        config.graph_options.optimizer_options.opt_level = -1

        with tf.Session(config = config) as sess:
            for device in ['/cpu:0']: #/GPU:0
                with tf.device(device):

                    d_input = tensor = tf.constant(1.0, shape=[20, 20, 10000], dtype=tf.float64)
                    d_W     = tensor = tf.constant(0.5, shape=[3, 3,   10000], dtype=tf.float64)

                    Wx_recurrent = recurrent_module.recurrent(d_input, d_W, 50)
                    r = sess.run(Wx_recurrent)
                    print(r)
                
if __name__ == '__main__':
    unittest.main()