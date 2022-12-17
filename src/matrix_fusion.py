#!/usr/bin/env python

import rospy
import numpy as np
import numpy.linalg as lin
import math
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov5_ros.msg import BoundingBoxes
from yolov5_ros.msg import BoundingBox
from yolov5_ros.msg import CameraObjectList
from yolov5_ros.msg import RadarObjectList

class fusion():
    def __init__(self):
        self.bridge = CvBridge()
        self.fusion_index_list = []
        self.fusion_distance_list = []
        self.camera_object_list = []
        self.radar_object_list = []
        self.bounding_box_list = []

        rospy.init_node('fusion_node', anonymous=False)
        # rospy.Subscriber('camera_objects', CameraObjectList, self.camera_object_callback)
        rospy.Subscriber('yolov5/detections', BoundingBoxes, self.camera_object_callback)
        rospy.Subscriber('radar_objects', RadarObjectList, self.radar_object_callback)
        rospy.Subscriber('/yolov5/image_out', Image, self.visualize)

    def camera_object_callback(self, data):
        self.bounding_box_list = data.bounding_boxes

    def radar_object_callback(self, data):
        self.radar_object_list = data.RadarObjectList
    
    def inverse_transform(self):
        vFOV = 2 * math.atan((0.5 * 960) / (0.5 * 1280 / math.tan(math.radians(25))))
        intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 480/math.tan(0.5*vFOV), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 640/math.tan(0.5*math.radians(40.9)), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 480/math.tan(0.5*math.radians(14)), 480], [0, 0, 1]])
        projection_matrix = np.array([[math.cos(math.radians(80)), -math.sin(math.radians(80)), 0, -2.3], [0, 0, -1, 1], [math.sin(math.radians(80)), math.cos(math.radians(80)), 0, 0.5]])
        
        inverse_matrix = lin.pinv(intrinsic_matrix @ projection_matrix)

        for bbox in self.bounding_box_list:
            x = (bbox.xmin + bbox.xmax) / 2
            y = (bbox.ymin + bbox.ymax) / 2

            image_matrix = np.array([[x], [y], [1]])
            
            print("------Result------")
            
            world_matrix = inverse_matrix @ image_matrix

            # print(world_matrix)

            scaling = world_matrix[3][0]

            world_matrix /= scaling

            # print(world_matrix)
            
            if len(self.radar_object_list) != 0:
                for radar_object in self.radar_object_list:
                    print("Transformed X value : ", world_matrix[0][0])
                    print("Radar X value : ", radar_object.x)
                    print("Transformed Y value : ", world_matrix[1][0])
                    print("Radar Y value : ", radar_object.y)
                    print("Transformed Z value : ", world_matrix[2][0])
                    print("Radar Z value : ", radar_object.z)
            
                    # transformed_matrix = intrinsic_matrix @ projection_matrix @ world_matrix

                    # z = transformed_matrix[2][0]

                    # print("Z value : ", z)
                    # print("Radar x value : ", radar_object.x)
                    

                # v_estimated = round(transform_estimated[0][0])
                # x_estimated = self.radar_object_list[min_idx].x
                # correction = (y - v_estimated) / 960 * x_estimated

                # x += correction

                # print("Corrected X value : ", x)
                
                
            # print("Y value : ", world_matrix[1][0])
            # print("Z value : ", world_matrix[2][0])
    
    def transform(self):
        # fovY = 2*math.atan(640*math.tan(math.radians(25)) / 480)
        vFOV = 2 * math.atan((0.5 * 960) / (0.5 * 1280 / math.tan(math.radians(25))))
        intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 640/math.tan(0.5*math.radians(50)), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(40.9)), 0, 640], [0, 480/math.tan(0.5*math.radians(31.3)), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]])
        projection_matrix = np.array([[math.cos(math.radians(80)), -math.sin(math.radians(80)), 0, -2.3], [0, 0, -1, 1], [math.sin(math.radians(80)), math.cos(math.radians(80)), 0, 0.5]])

        for radar_object in self.radar_object_list:

            world_point = np.array([[radar_object.x], [radar_object.y], [radar_object.z], [1]])

            print("X : ", radar_object.x)

            transformed_matrix = intrinsic_matrix @ projection_matrix @ world_point

            scaling = transformed_matrix[2][0]

            transformed_matrix /= scaling

            # print("Float Point : ", transformed_matrix[0][0], transformed_matrix[1][0])
            x = round(transformed_matrix[0][0])
            y = round(transformed_matrix[1][0])

            print("Integer Point : ", x, y)

            cv2.line(self.image, (x,y), (x,y), (0, 0, 255), 20)

    def transformation_demo(self):
        # intrinsic_matrix = np.array([[2400, 0, 640], [0, 2400, 480], [0, 0, 1]])

        # Rt = np.array([[math.cos(math.radians(80)), -math.sin(math.radians(80)), 0], [0, 0, -1], [math.sin(math.radians(80)), math.cos(math.radians(80)), 0]]).T
        Rt = np.array([[math.cos(math.radians(80)), 0, math.sin(math.radians(80))], [-math.sin(math.radians(80)), 0, math.cos(math.radians(80))], [0, -1, 0]])
        
        for bbox in self.bounding_box_list:
            x = (bbox.xmin + bbox.xmax) / 2
            y = bbox.ymax

            # fovY = 2 * math.atan(640*math.tan(math.radians(25)) / 480)
            fx = 640/math.tan(math.radians(40.9))
            fy = 480/math.tan(math.radians(31.3))

            u = (x - 640) / fx
            v = (y - 480) / fy

            Pc = np.array([[u], [v], [1]])

            t = np.array([[-2.3], [1], [0.5]])

            pw = np.dot(Rt, (Pc-t))
            cw = np.dot(Rt, (-t))

            k = (cw[2][0] + 0.5) / (cw[2][0] - pw[2][0])

            world_point = cw + k*(pw-cw)
            print("=============================================================")
            if len(self.radar_object_list) != 0:
                for radar_object in self.radar_object_list:
                    # print("x point : ", world_point[0][0])
                    # print("y point : ", world_point[1][0])
                    # print("z point : ", world_point[2][0])
                    print(world_point)
                    print("----------------------------------")
                    print("radar x : ", radar_object.x)
                    print("radar y : ", radar_object.y)
                    print("radar z : ", radar_object.z)
                    print("-------------------------------------------------------------")

    def visualize(self, data):

        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # print(self.image.shape)

        
        # self.inverse_transform()
        self.transform()
        # self.transformation_demo()

        cv2.imshow("Display", self.image)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        fusion()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
