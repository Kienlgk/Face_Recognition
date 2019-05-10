import cv2

def test_cam():
    # cap = cv2.VideoCapture("http://admin:123456789@192.168.1.12:81/videostream.cgi?action=stream&dummy=param.mjpg")
    # cap.open("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789?action=stream?dummy=param.mjpg")
    cap = cv2.VideoCapture("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789")
    # cap.open("http://192.168.1.12:81/videostream.cgi?user=admin&pwd=123456789")
    # cap = cv2.VideoCapture('/home/na/workspace/hdd/Downloads/sample.mp4')
    print(cap.isOpened())
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    cap.release()

if __name__ == "__main__":
    test_cam()