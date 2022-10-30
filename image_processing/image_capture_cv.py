import cv2
import os
from time import time, ctime

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory." + directory)


class image_capture:
    def __init__(self, frame_divide):
        self.img_cnt = 0
        self.cnt = 0
        self.folder_name = "/home/heven/CoDeep_ws/src/yolov5_ros/image_processing/" + str(ctime(time()))
        self.video_path = "/home/heven/CoDeep_ws/src/yolov5_ros/image_processing/video/test1.mp4"
        createFolder(self.folder_name)
        self.capture(frame_divide)
    
    def capture(self, frame_divide):
        vidcap = cv2.VideoCapture(self.video_path)
        while (vidcap.isOpened()):
            ret, img = vidcap.read()

            if not ret:
                print("No video")
                break

            if self.img_cnt % frame_divide == 0:
                photo_name = self.folder_name + "/" + str(self.cnt) + ".jpg"
                cv2.imwrite(photo_name, img)
                print("Photo %d is saved." % (self.cnt))
                self.cnt += 1

            self.img_cnt += 1


if __name__ == '__main__':
    frame_divide = 5
    ic = image_capture(frame_divide)
    
    print("-----Starting Image Capture-----")
    print("Extract 1 picture from %d frames", frame_divide)
    
    cv2.destroyAllWindows()