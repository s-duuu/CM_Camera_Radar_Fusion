#!/usr/bin/env python2

import cv2
import os
import rospy
from time import time, ctime
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory." + directory)


class image_capture:
    def __init__(self):
        self.subscriber = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback, queue_size = 1)
        self.bridge = CvBridge()
        self.img_cnt = 0
        self.cnt = 0
        self.folder_name = "/home/heven/CoDeep_ws/src/YOLO_Project/yolo_codeep/photo/" + str(ctime(time()))
        createFolder(self.folder_name)
    
    def callback(self, ros_data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(ros_data, "bgr8")
            cv2.imshow("img", cv_image)
            cv2.waitKey(1)

            if self.img_cnt % 10 == 0:
                photo_name = self.folder_name + "/" + str(self.cnt) + ".jpg"
                cv2.imwrite(photo_name, cv_image)
                rospy.loginfo("Photo %d is saved." % (self.cnt))
                self.cnt += 1

            self.img_cnt += 1

        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    ic = image_capture()
    rospy.init_node('image_capture', anonymous=True)
    rospy.loginfo("now starts yolo node...")
    
    try:
        rospy.spin()
    
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()