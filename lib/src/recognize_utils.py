from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
from six import iteritems
import numpy as np
from datetime import datetime
import pickle
import time
import argparse
import cv2
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc

import tensorflow as tf
import cv2 
from 
def recognize_face_stream(sess, pnet, rnet, onet, feature_array, args):
    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = args.image_size
    embedding_size = embeddings.get_shape()[1]
    
    # cap = cv2.VideoCapture("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789?action=stream?dummy=param.mjpg")
    cap = cv2.VideoCapture("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789")

    # cap.open("192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789&type=some.mjpeg")
    i = 0
    while(True):
        ret, frame = cap.read()
        cap_time = datetime.now()
        if not ret:
            print("Error: failed to capture image")
            break
        gray = cv2.cvtColor(frame, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if (gray.size > 0):
            start = time.time()
            response, faces,bboxs = align_face(gray,pnet, rnet, onet, args)
            print(response)
            if (response == True):
                cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam', '1', cap_time.strftime("%m_%d_%Y_%H_%M__S")+".png")), gray)
                for i, image in enumerate(faces):
                    bb = bboxs[i]
                    images = load_img(image, False, False, image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    result, accuracy = identify_person(feature_vector, feature_array, 8)
                    print(result.split("/")[1])
                    print(accuracy)

                    if accuracy < 0.9:
                        cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                        W = int(bb[2]-bb[0])//2
                        H = int(bb[3]-bb[1])//2
                        cv2.putText(gray,"Hello "+result.split("/")[1],(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    else:
                        cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                        W = int(bb[2]-bb[0])//2
                        H = int(bb[3]-bb[1])//2
                        cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    # cv2.imsave(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'cam', '1', str(i)+".png")), gray)
                    del feature_vector
            stop = time.time()
            print('Cost {} s for embedding 1 image'.format(stop - start))
            cv2.imshow('img',gray)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(i)
            i += 1
        else:
            continue
    cap.release()
    pass
