from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from tensorflow.python.platform import gfile

def main(args):
  
    # with tf.Graph().as_default():
      
    #     with tf.Session() as sess:
    #         pass
    with tf.Session() as sess:
        model_filename = args.model
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        # [print(v.name) for v in tf.global_variables()]
        graph = tf.get_default_graph()
        list_of_tuples = [op.values()[0] for op in graph.get_operations()]
        [print(tensor) for tensor in list_of_tuples]
        
    LOGDIR='~/log/facenet/pretrained'
    # [print(n.name, n.shape) for n in tf.get_default_graph().as_graph_def().node]
    # [print(op.name, op.values) for op in tf.get_default_graph().get_operations()]
    # train_writer = tf.summary.FileWriter(os.path.expanduser(LOGDIR))
    # train_writer.add_graph(sess.graph)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/media/na/e03497de-87fb-4f35-a524-811dfc5997d9/workspace/Kien/Face_Recognition/lib/src/ckpt/20180402-114759/20180402-114759.pb', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))