import os
import argparse
import base64
import numpy as np
import cv2
import threading
import queue
from scipy import misc



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

    def __del__(self):
        self.cap.release()
        self.q.get_nowait()


global_cap = VideoCapture('rtsp://admin:KtxBk$2019@222.253.145.118:55431/h264/ch1/main/av_stream')


def read_stream(args=None):
    ext = ".png"
    # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Video', 960, 540)
    while True:
        frame = global_cap.read()
        frame = misc.imresize(frame, (540, 960), interp='bilinear')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if (frame.size > 0):
            cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            continue
    pass


def main():
    read_stream()


if __name__ == '__main__':
    main()
