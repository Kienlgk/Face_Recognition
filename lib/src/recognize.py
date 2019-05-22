from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
from six import iteritems
import numpy as np
from datetime import datetime
from imutils.video import FPS
import threading

# from threading import Lock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

# import retrieve
from align import detect_face
import tensorflow as tf
import pickle
import time
import argparse
import base64
import cv2
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc

from facenet import load_data
from facenet import load_img
from facenet import load_model
from facenet import to_rgb
from facenet import get_model_filenames

import queue


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        # cv2.VideoCapture.set(self.cap, cv2.CAP_PROP_CONVERT_RGB, 1)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def get_base64(self):
        return base64.b64encode(self.q.get().tobytes())


class ReturnQueue:
    def __init__(self):
        self.q = queue.Queue()

    def add(self, frame_base64):
        if not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(frame_base64)

    def get(self):
        if not self.q.empty():
            try:
                return self.q.get()
            except queue.Empty:
                pass
        return None


def main(args):
    ckpt = os.path.expanduser(args.ckpt)
    embedding_dir = os.path.expanduser(args.embedding_dir)
    is_stream = args.stream
    gpu_memory_fraction = args.gpu_memory_fraction
    graph_fr = tf.Graph()
    # Config memory for GTX 1080 with 8GB VRAM
    gpu_memory_fraction = 0.4
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # sess = tf.Session(config=config)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_fr = tf.Session(graph=graph_fr, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    meta_graph_file, ckpt_file = get_model_filenames(ckpt)
    with graph_fr.as_default():
        saverf = tf.train.import_meta_graph(os.path.join(ckpt, meta_graph_file))
        saverf.restore(sess_fr, os.path.join(ckpt, ckpt_file))
        pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)

    with open(embedding_dir, 'rb') as f:
        embedding_dict = pickle.load(f)

    def face_det(is_stream):
        if is_stream:
            recognize_face_stream(sess_fr, pnet, rnet, onet, feature_array=embedding_dict, args=args)
        else:
            recognize_face(sess_fr, pnet, rnet, onet, feature_array=embedding_dict, args=args)

    face_det(is_stream)


def recognize_face(sess, pnet, rnet, onet, feature_array, args):
    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder, (160, 160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    image_size = args.image_size
    imgdir = os.path.expanduser(args.imgdir)
    '''
    if os.path.isfile(imgdir):
        gray = cv2.imread(imgdir, cv2.IMREAD_GRAYSCALE)
        if(gray.size > 0):
            print(gray.size)
            start = time.time()
            response, faces,bboxs = align_face(gray,pnet, rnet, onet, args)
            align = time.time()
            print(response)
            if (response == True):
                for i, image in enumerate(faces):
                    bb = bboxs[i]
                    images = load_img(image, False, False, image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    result, accuracy = identify_person(feature_vector, feature_array, 8)
                    print(result.split("/")[1])
                    print(accuracy)

                    # if accuracy < 0.9:
                    #     cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                    #     W = int(bb[2]-bb[0])//2
                    #     H = int(bb[3]-bb[1])//2
                    #     cv2.putText(gray,result.split("/")[1],(bb[0]+W-(W//2),bb[1]-7), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    # else:
                    #     cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                    #     W = int(bb[2]-bb[0])//2
                    #     H = int(bb[3]-bb[1])//2
                    #     cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,
                    (255,255,255),1,cv2.LINE_AA)
                    del feature_vector
            stop = time.time()
            print('Cost {} s for embedding 1 image'.format(stop - start))
            print('Cost {} s for align 1 image'.format(align-start))
            # cv2.imshow('img',gray)
            cv2.waitKey(0)
        return
    '''
    image_dirs = os.listdir(imgdir)
    for image_dir in image_dirs:
        image_dir = os.path.join(imgdir, image_dir)
        # gray = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
        gray = cv2.imread(image_dir)
        # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if (gray.size > 0):
            start = time.time()
            response, faces, bboxs = align_face(gray, pnet, rnet, onet, args)
            print(response)
            if (response == True):
                for i, image in enumerate(faces):
                    bb = bboxs[i]
                    cv2.rectangle(gray, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 2)
                    W = int(bb[2] - bb[0]) // 2
                    H = int(bb[3] - bb[1]) // 2
                    cv2.putText(gray, "Face", (bb[0] + W - (W // 2), bb[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    '''
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
                        cv2.putText(gray, result.split("/")[1],(bb[0]+W-(W//2),bb[1]-7), 
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    else:
                        cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                        W = int(bb[2]-bb[0])//2
                        H = int(bb[3]-bb[1])//2
                        cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,
                        255,255),1,cv2.LINE_AA)
                    del feature_vector
                    '''
                cv2.imwrite(os.path.expanduser(
                    os.path.join('~', 'workspace', 'hdd', 'Kien', 'data', 'cam', 'tuong_test_aligned', 'tuong_1_reg',
                                 'tuong_face', os.path.split(image_dir)[-1])), gray)
            else:
                cv2.imwrite(os.path.expanduser(
                    os.path.join('~', 'workspace', 'hdd', 'Kien', 'data', 'cam', 'tuong_test_aligned', 'tuong_1_reg',
                                 'tuong_no_face', os.path.split(image_dir)[-1])), gray)

            stop = time.time()
            print('Cost {} s for embedding 1 image'.format(stop - start))
            cv2.imshow('img', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # cv2.waitKey(0)
        else:
            continue


# lo = Lock()
latest_frame = None
last_ret = None

global_cap = VideoCapture("")
global_return_queue = ReturnQueue()


def get_frame():
    return global_cap.get_base64()


def recognize_face_stream(sess, pnet, rnet, onet, feature_array, args):
    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder, (160, 160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = args.image_size
    embedding_size = embeddings.get_shape()[1]

    # cap = cv2.VideoCapture("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789?action=stream?dummy
    # =param.mjpg")
    # cap = cv2.VideoCapture("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789")
    # cap = VideoCapture("")
    # def rtsp_cam_buffer(vcap):
    #     global latest_frame, last_ret
    #     while True:
    #         last_ret, latest_frame = vcap.read()

    # t1 = threading.Thread(target=rtsp_cam_buffer,args=(cap, ),name="rtsp_read_thread")
    # t1.daemon=True
    # t1.start()
    # fps = FPS().start()

    threshold = 0.9
    ext = ".png"
    # cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Video', 960, 540)
    while (True):
        # if (last_ret is not None) and (latest_frame is not None):
        #     frame = latest_frame.copy()
        # else:
        #     print("Error: failed to capture image")
        #     break
        frame = global_cap.read()
        # ret, frame = cap.read()
        cap_time = datetime.now()
        # frame = misc.imread(frame)
        # frame = cv2.cvtColor(frame, 0)
        # gray = cv2.cvtColor(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gray = gray[384:, 216:]
        # gray = cv2.resize(gray, (960, 540))
        # gray = cv2.resize(gray, (960, 540))
        # gray = cv2.resize(frame, (960, 540))
        # frame = cv2.resize(frame, (960, 540))
        gray = misc.imresize(gray, (540, 960), interp='bilinear')
        frame = misc.imresize(frame, (540, 960), interp='bilinear')

        # gray = cv2.resize(gray, (768, 432))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if (gray.size > 0):
            start = time.time()
            response, faces, bboxs = align_face(gray, pnet, rnet, onet, args)
            align_face_time = time.time()
            print(response)
            if (response == True):
                # cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam', '1',
                # cap_time.strftime("%m_%d_%Y_%H_%M__S")+".png")), gray)
                # cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam',
                # 'tuong_1_collect', str(cap_time)+".png")), frame)
                # cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam',
                # 'tuong_test', '0001', 'tuong_5_raw', str(cap_time)+ext)), frame)

                for i, image in enumerate(faces):
                    bb = bboxs[i]
                    images = load_img(image, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    # detect face only
                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 2)
                    W = int(bb[2] - bb[0]) // 2
                    H = int(bb[3] - bb[1]) // 2
                    # cv2.putText(frame, "Face", (bb[0] + W - (W // 2), bb[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (255, 255, 255), 1, cv2.LINE_AA)
                    # Calculate face embedding and compare with trained ones
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    result, accuracy = identify_person(feature_vector, feature_array, 8)
                    print(result.split("/")[-2])
                    print(accuracy)
                    if accuracy < threshold:
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 2)
                        W = int(bb[2] - bb[0]) // 2
                        H = int(bb[3] - bb[1]) // 2
                        cv2.putText(frame, result.split("/")[-2] + " dist: " + str(accuracy),
                                    (bb[0] + W - (W // 2), bb[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                    1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 2)
                        W = int(bb[2] - bb[0]) // 2
                        H = int(bb[3] - bb[1]) // 2
                        cv2.putText(frame, "?????", (bb[0] + W - (W // 2), bb[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    del feature_vector
                # cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam', 'vom',
                # str(cap_time)+".png")), gray)
                # cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam', 'tuong',
                # str(cap_time)+".png")), frame)
                # cv2.imwrite(os.path.expanduser(os.path.join('~','workspace','hdd','Kien', 'data', 'cam',
                # 'tuong_1_collect', str(cap_time)+".png")), frame)
                cv2.imwrite(os.path.expanduser(
                    os.path.join('~', 'workspace', 'hdd', 'Kien', 'data', 'demo', 'tuong', str(cap_time) + ext)), frame)

            stop = time.time()
            print('Cost {} s for aligning 1 image'.format(align_face_time - start))
            print('Cost {} s for matching embedding 1 image'.format(stop - start))
            # cv2.imshow('img', cv2.cvtColor(gray, cv2.COLOR_RGB2BGR))
            # cv2.imshow('Video', frame)
            base64_frame = base64.b64encode(frame.tobytes())
            global_return_queue.add(base64_frame)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            continue
    cap.release()
    pass


def align_face(img, pnet, rnet, onet, args):
    minsize = 30  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    if img.size == 0:
        print("empty array")
        return False, img, [0, 0, 0, 0]

    if img.ndim < 2:
        print('Unable to align')

    if img.ndim == 2:
        img = to_rgb(img)

    img = img[:, :, 0:3]

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces == 0:
        return False, img, [0, 0, 0, 0]
    else:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if args.detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        if len(det_arr) > 0:
            faces = []
            bboxes = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
            # misc.imsave("cropped.png", scaled)
            faces.append(scaled)
            bboxes.append(bb)
        return True, faces, bboxes


def identify_person(image_vector, feature_array, k=9):
    # features = np.asarray(list(feature_array.values()))
    # noEmbedding = features.shape[0]
    # image_vector = np.tile(image_vector, (noEmbedding, 1))
    # top_k_ind = np.argsort(np.linalg.norm(image_vector-features, axis=1))
    top_k_ind = np.argsort([np.linalg.norm(image_vector - pred_row) for pred_row in (list(feature_array.values()))])[:k]
    # for ith_row, pred_row in enumerate(list(feature_array.values()))])[:k]

    result = list(feature_array.keys())[top_k_ind[0]]
    print("result: ", result)
    acc = np.linalg.norm(image_vector - list(feature_array.values())[top_k_ind[0]])
    return result, acc


def arguments_parser(argv):
    handler = argparse.ArgumentParser()
    handler.add_argument("--imgdir", default="", help="Testing images directory")
    handler.add_argument("--embedding_dir",
                         default="~/workspace/hdd/Kien/Face_Recognition/lib/src/embedding_ktx.pickle",
                         help="Directory of the embedding file")
    handler.add_argument("--ckpt", default="~/workspace/hdd/Kien/Face_Recognition/lib/src/ckpt/20180402-114759",
                         help="Model's weights .ckpt file")
    handler.add_argument("--gpu_memory_fraction", default=0.85)
    handler.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    handler.add_argument('--detect_multiple_faces', type=bool, help='Detect and align multiple faces per image.',
                         default=True)
    handler.add_argument('--margin', type=int,
                         help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    handler.add_argument("--stream", default=False, action='store_true')
    return handler.parse_args()


if __name__ == "__main__":
    main(arguments_parser(sys.argv[1:]))
