#!flask/bin/python
################################################################################################################################

# ------------------------------------------------------------------------------------------------------------------------------
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end
# reaches
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #

# -------------------------------------------------------------------------------------------------------------------------------
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for, redirect, render_template
# from flask.ext.httpauth import HTTPBasicAuth
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import sys
import random
from tensorflow.python.platform import gfile
from six import iteritems
import tensorflow as tf
import numpy as np
import pickle
import time

# sys.path.append('..')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.src import retrieve
from lib.src.align import detect_face
from tensorflow.python.platform import gfile
from lib.src.facenet import get_model_filenames
from lib.src.recognize import recognize_face_stream
from lib.src.recognize import get_frame

from dict2obj import to_obj

app = Flask(__name__, static_url_path="")

auth = HTTPBasicAuth()

config_args = {"detect_multiple_faces": True, "margin": 44, "image_size": 160,
               "embedding_dir": "~/workspace/hdd/Kien/Face_Recognition/lib/src/embedding_ktx.pickle",
               "ckpt": "~/workspace/hdd/Kien/Face_Recognition/lib/src/ckpt/20180402-114759"}

# ==============================================================================================================================
#

#    Loading the stored face embedding vectors for image retrieval

#                                                                          						        
#

# ==============================================================================================================================
EMBEDDING_DIR = os.path.expanduser(config_args["embedding_dir"])
CKPT = os.path.expanduser(config_args["ckpt"])

with open(EMBEDDING_DIR, 'rb') as f:
    feature_array = pickle.load(f)

ckpt = CKPT

graph_fr = tf.Graph()

gpu_options = tf.GPUOptions(allow_growth=True)
sess_fr = tf.Session(graph=graph_fr, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
meta_graph_file, ckpt_file = get_model_filenames(ckpt)
with graph_fr.as_default():
    saverf = tf.train.import_meta_graph(os.path.join(ckpt, meta_graph_file))
    saverf.restore(sess_fr, os.path.join(ckpt, ckpt_file))
    pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)

# ==============================================================================================================================
#

#  This function is used to do the face recognition from video camera

#                                                                                                 
#

# ==============================================================================================================================
# recognize_face_stream(sess_fr, pnet, rnet, onet, feature_array, to_obj(config_args))


@app.route('/facerecognitionLive', methods=['GET', 'POST'])
def face_det():
    # recognize_face_stream(sess_fr, pnet, rnet, onet, feature_array, to_obj(config_args))
    return get_frame()
#
# @app.route('/getLatestFrame', method=['GET', 'POST'])
# def get_latest_frame():
#     return get_frame()


# ==============================================================================================================================
#

#                                           Main function
#                                           #
#  				                                                                                                
# ==============================================================================================================================
@app.route("/")
def main():
    return render_template("main.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8888')
