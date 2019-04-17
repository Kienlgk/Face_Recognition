from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
from six import iteritems
import numpy as np
import retrieve
from align import detect_face
import tensorflow as tf
import pickle
import time
import argparse

def main(args):
    ckpt = args.ckpt
    embedding_dir = args.embedding_dir
    gpu_memory_fraction = args.gpu_memory_fraction
    graph_fr = tf.Graph()
    # Config memory for GTX 1080 with 8GB VRAM
    gpu_memory_fraction = 0.4
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess_fr = tf.Session(graph=graph_fr, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with graph_fr.as_default():
        saverf = tf.train.import_meta_graph(os.path.join(ckpt, 'model-20180402-114759.meta'))
        saverf.restore(sess_fr, os.path.join(ckpt, 'model-20180402-114759.ckpt-275'))
        pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)

    with open(embedding_dir, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    def face_det():
        retrieve.recognize_face(sess_fr, pnet, rnet, onet, embedding_dict)

    face_det()


def recognize_face(sess,pnet, rnet, onet,feature_array):
    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = args.image_size
    embedding_size = embeddings.get_shape()[1]

    cap = cv2.VideoCapture(-1)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if(gray.size > 0):
            print(gray.size)
            response, faces,bboxs = align_face(gray,pnet, rnet, onet)
            print(response)
            if (response == True):
                    for i, image in enumerate(faces):
                            bb = bboxs[i]
                            images = load_img(image, False, False, image_size)
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                            result, accuracy = identify_person(feature_vector, feature_array,8)
                            print(result.split("/")[2])
                            print(accuracy)

                            if accuracy < 0.9:
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                cv2.putText(gray,"Hello "+result.split("/")[2],(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                            del feature_vector

            cv2.imshow('img',gray)
        else:
            continue


def align_face(img,pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print("before img.size == 0")	
    if img.size == 0:
        print("empty array")
        return False,img,[0,0,0,0]

    if img.ndim<2:
        print('Unable to align')

    if img.ndim == 2:
        img = to_rgb(img)

    img = img[:,:,0:3]

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]

            
    if nrof_faces==0:
        return False,img,[0,0,0,0]
    else:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            if args.detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))
        if len(det_arr)>0:
                faces = []
                bboxes = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-args.margin/2, 0)
            bb[1] = np.maximum(det[1]-args.margin/2, 0)
            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
            misc.imsave("cropped.png", scaled)
            faces.append(scaled)
            bboxes.append(bb)
            print("leaving align face")
        return True,faces,bboxes


def identify_person(image_vector, feature_array, k=9):
    top_k_ind = np.argsort([np.linalg.norm(image_vector-pred_row) \
                        for ith_row, pred_row in enumerate(feature_array.values())])[:k]
    result = feature_array.keys()[top_k_ind[0]]
    acc = np.linalg.norm(image_vector-feature_array.values()[top_k_ind[0]])
    return result, acc


if __name__ == "__main__":
    handler = argparse.ArgumentParser()
    handler.add_argument("--imgdir", help="Testing images directory")
    handler.add_argument("--embedding_dir", help="Directory of the embedding file")
    handler.add_argument("--ckpt", help="Model's weights .ckpt file")
    handler.add_argument("--gpu_memory_fraction", default=0.85)

    # handler.add_argument()
    FLAGS = handler.parse_args()
    main(FLAGS)