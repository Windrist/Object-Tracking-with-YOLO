import time
import cv2
from threading import Thread
import math

from utils import detect_image, load_yolo_model
from configs import *


class ObjectTracker(object):
    def __init__(self):
        self.yolo = load_yolo_model()
        self.vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        # self.vid.set(3, 1920)
        # self.vid.set(4, 1080)

        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'H264')
        self.out = cv2.VideoWriter('Output/Out.avi', codec, 25, (width, height))

        self.yolo_time = time.time()
        self.tracker_time = time.time()
        self.tracker = None
        self.box = None

        self.od_times = []
        self.ot_times = []
        self.stop = False
        self.got = False
        self.ret, self.frame = self.vid.read()

    def yolo_thread(self):
        count = 0
        while True:
            self.yolo_time = time.time()
            ret, frame = self.vid.read()
            if not ret:
                break
            if not self.got:
                self.box = detect_image(self.yolo, frame, input_size=YOLO_INPUT_SIZE)
                if self.box is not None:
                    self.got = True
            else:
                box = detect_image(self.yolo, frame, input_size=YOLO_INPUT_SIZE)
                if box is None:
                    count = count + 1
                else:
                    dis_x = (box[0] + box[2] / 2) - (self.box[0] + self.box[2] / 2)
                    dis_y = (box[1] + box[3] / 2) - (self.box[1] + self.box[3] / 2)
                    dis = math.sqrt(dis_x*dis_x + dis_y*dis_y)
                    if dis > 100:
                        count = count + 1
                    else:
                        count = 0
                if count > 5:
                    count = 0
                    self.got = False
                    self.tracker = None
            self.od_times.append(time.time() - self.yolo_time)
            self.od_times = self.od_times[-20:]
            s = sum(self.od_times) / len(self.od_times)
            print("YOLO FPS: ", 1 / s)
            if self.stop:
                break

    def tracker_thread(self):
        while True:
            self.tracker_time = time.time()
            ret, frame = self.vid.read()
            if not ret:
                break
            if self.got:
                prev_time = time.time()
                if self.tracker is None:
                    if TRACKER_TYPE == "MOSSE":
                        self.tracker = cv2.legacy.TrackerMOSSE_create()
                    elif TRACKER_TYPE == "KCF":
                        self.tracker = cv2.TrackerKCF_create()
                    elif TRACKER_TYPE == "CSRT":
                        self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, self.box)
                reto, self.box = self.tracker.update(frame)

                if reto:
                    # Draw New Box of Tracked Object
                    tl, br = (int(self.box[0]), int(self.box[1])), (
                        int(self.box[0] + self.box[2]), int(self.box[1] + self.box[3]))
                    cv2.rectangle(frame, tl, br, (0, 255, 0), 2, 2)

                    # Check FPS
                    if time.time() - prev_time > 0:
                        self.ot_times.append(time.time() - prev_time)
                        self.ot_times = self.ot_times[-20:]
                        s = sum(self.ot_times) / len(self.ot_times)
                        cv2.putText(frame, "Time: {:.1f}FPS".format(1 / s), (0, 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                else:
                    # Print Failed
                    self.got = False
                    self.tracker = None
                    cv2.putText(frame, "Tracker Update Failure!", (0, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Waiting to Detect Object...", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            # Show Video from Webcam
            cv2.imshow("Video", frame)
            self.out.write(frame)

            # Detect Key
            key = cv2.waitKey(1) & 0xff

            # Exit if Press Q Key
            if key == ord('q'):
                self.stop = True
                break

    def start(self):
        Thread(target=self.yolo_thread).start()
        Thread(target=self.tracker_thread).start()


if __name__ == '__main__':
    track = ObjectTracker()
    track.start()
