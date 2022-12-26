#!/usr/bin/env python

import rospy
import time
import os
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import cv2
import pandas as pd

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
        self.radar_object_list = []
        self.filtered_radar_object_list = []
        self.bounding_box_list = []
        self.distance_thresh = 6
        self.angle_thresh = 30
        self.my_speed = 20

        rospy.init_node('fusion_node', anonymous=False)
        # rospy.Subscriber('camera_objects', CameraObjectList, self.camera_object_callback)
        rospy.Subscriber('yolov5/detections', BoundingBoxes, self.camera_object_callback)
        rospy.Subscriber('radar_objects', RadarObjectList, self.radar_object_callback)
        rospy.Subscriber('/yolov5/image_out', Image, self.visualize)

    def camera_object_callback(self, data):
        now = rospy.get_rostime()
        rospy.loginfo("Time : %i", now.secs)
        self.bounding_box_list = data.bounding_boxes

    def radar_object_callback(self, data):
        self.radar_object_list = data.RadarObjectList
    
    def is_in_bbox(self, bbox, radar_2d):
        
        if radar_2d[0] > bbox.xmin and radar_2d[0] < bbox.xmax and radar_2d[1] > bbox.ymin and radar_2d[1] < bbox.ymax:
            return True
        
        else:
            return False

    def inverse_transform(self):
        # vFOV = 2 * math.atan((0.5 * 960) / (0.5 * 1280 / math.tan(math.radians(25))))
        intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 640/math.tan(0.5*math.radians(50)), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 640/math.tan(0.5*math.radians(40.9)), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 480/math.tan(0.5*math.radians(14)), 480], [0, 0, 1]])
        projection_matrix = np.array([[0.1736, -0.9848, 0, 1.3842], [0, 0, -1, 0.5], [0.9848, 0.1736, 0, 2.0914]])
        
        inverse_matrix = lin.pinv(intrinsic_matrix @ projection_matrix)

        for bbox in self.bounding_box_list:
            x = (bbox.xmin + bbox.xmax) / 2
            y = (bbox.ymin + bbox.ymax) / 2

            image_matrix = np.array([[x], [y], [1]])
            
            print("====================================================================")
            
            world_matrix = inverse_matrix @ image_matrix

            # print(world_matrix)

            scaling = world_matrix[3][0]

            # world_matrix /= scaling

            print(world_matrix)
            
            if len(self.radar_object_list) != 0:
                for radar_object in self.radar_object_list:
                    print("Transformed X value : ", world_matrix[0][0])
                    print("Transformed Y value : ", world_matrix[1][0])
                    print("Transformed Z value : ", world_matrix[2][0])
                    print("--------------------------------")
                    print("Radar X value : ", radar_object.x)
                    print("Radar Y value : ", radar_object.y)
                    print("Radar Z value : ", radar_object.z)
                    print("-------------------------------------------------------------")
            
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
    
    def transform(self, radar_object):
        # fovY = 2*math.atan(640*math.tan(math.radians(25)) / 480)
        # vFOV = 2 * math.atan((0.5 * 960) / (0.5 * 1280 / math.tan(math.radians(25))))
        # intrinsic_matrix = np.array([[800/math.tan(math.radians(25)), 0, 640], [0, 800/math.tan(math.radians(25)), 480], [0, 0, 1]])
        # intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(40.9)), 0, 640], [0, 480/math.tan(0.5*math.radians(31.3)), 480], [0, 0, 1]])
        intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 640/math.tan(0.5*math.radians(50)), 480], [0, 0, 1]])
        # projection_matrix = np.array([[math.cos(math.radians(80)), -math.sin(math.radians(80)), 0, -2.3], [0, 0, -1, 1], [math.sin(math.radians(80)), math.cos(math.radians(80)), 0, 0.5]])
        projection_matrix = np.array([[0.1736, -0.9848, 0, 1.3842], [0, 0, -1, 0.5], [0.9848, 0.1736, 0, 2.0914]])

        # print("=================================")
        
        world_point = np.array([[radar_object.x], [radar_object.y], [radar_object.z], [1]])

        transformed_matrix = intrinsic_matrix @ projection_matrix @ world_point

        scaling = transformed_matrix[2][0]

        transformed_matrix /= scaling

        x = round(transformed_matrix[0][0])
        y = round(transformed_matrix[1][0])
        
        # cv2.line(self.image, (x,y), (x,y), (0, 255, 0), 15)

        return (x,y)
            

    def transformation_demo(self):
        # intrinsic_matrix = np.array([[2400, 0, 640], [0, 2400, 480], [0, 0, 1]])

        # Rt = np.array([[math.cos(math.radians(80)), -math.sin(math.radians(80)), 0], [0, 0, -1], [math.sin(math.radians(80)), math.cos(math.radians(80)), 0]]).T
        Rt = np.array([[0.1736, -0.9848, 0], [0, 0, -1], [0.9848, 0.1736, 0]]).T
        
        for bbox in self.bounding_box_list:
            x = (bbox.xmin + bbox.xmax) / 2
            y = bbox.ymax

            # fovY = 2 * math.atan(640*math.tan(math.radians(25)) / 480)
            fx = 640/math.tan(math.radians(25))
            fy = fx

            u = (x - 640) / fx
            v = (y - 480) / fy

            Pc = np.array([[u], [v], [1]])

            t = np.array([[1.3842], [0.5], [2.0914]])

            pw = Rt @ (Pc-t)
            cw = Rt @ (-t)

            k = (cw[2][0] + 0.5) / (cw[2][0] - pw[2][0])

            world_point = cw + k*(pw-cw)
            # print("=============================================================")
            # if len(self.radar_object_list) != 0:
            #     for radar_object in self.radar_object_list:
            #         # print("x point : ", world_point[0][0])
            #         # print("y point : ", world_point[1][0])
            #         # print("z point : ", world_point[2][0])
            #         print(world_point)
            #         print("----------------------------------")
            #         print("radar x : ", radar_object.x)
            #         print("radar y : ", radar_object.y)
            #         print("radar z : ", radar_object.z)
            #         print("-------------------------------------------------------------")

            x_c = world_point[0][0]
            y_c = world_point[1][0]

            camera_object = (x_c, y_c)

            # self.first_matching(camera_object)
            self.second_matching(bbox, camera_object)
            

    def first_matching(self, camera_object):

        if len(self.radar_object_list) != 0:
            for radar_object in self.radar_object_list:
                cv2.line(self.image, self.transform(radar_object), self.transform(radar_object), (0, 255, 0), 15)
                # print("point distance : ", (math.sqrt(((camera_object[0] - radar_object.x)**2) + ((camera_object[1] - radar_object.y)**2))))
                # print("point angle : ", math.degrees(abs(math.atan(camera_object[0]/camera_object[1]) - math.atan(radar_object.x/radar_object.y))))
                if (math.sqrt(((camera_object[0] - radar_object.x)**2) + ((camera_object[1] - radar_object.y)**2)) < self.distance_thresh) and math.degrees(abs(math.atan(camera_object[0]/camera_object[1]) - math.atan(radar_object.x/radar_object.y))) < self.angle_thresh:
                    self.filtered_radar_object_list.append(radar_object)

        min_iou = math.inf
        min_idx = 0
        cnt = 0

        if len(self.filtered_radar_object_list) != 0:
            for radar_object in self.filtered_radar_object_list:
                
                if math.sqrt((self.transform(radar_object)[0] - camera_object[0])**2 + (self.transform(radar_object)[1] - camera_object[1])**2) < min_iou:
                    min_idx = cnt
                    min_iou = math.sqrt((self.transform(radar_object)[0] - camera_object[0])**2 + (self.transform(radar_object)[1] - camera_object[1])**2)
                
                cnt += 1
        
            min_x = self.transform(self.filtered_radar_object_list[min_idx])[0]
            min_y = self.transform(self.filtered_radar_object_list[min_idx])[1]
        
            cv2.line(self.image, (min_x, min_y), (min_x, min_y), (0, 255, 0), 15)
        
            final_distance = camera_object[0]
            final_velocity = self.filtered_radar_object_list[min_idx].velocity
            
            print("Real distance : 15.52m")
            print("Distance : ", final_distance)
            # print("Radar distance : ", self.filtered_radar_object_list[min_idx].x)
            print("Velocity : ", final_velocity * 3.6)

            self.risk_calculate(final_distance, final_velocity)
    
    def second_matching(self, bbox, camera_object):
        
        
        if len(self.radar_object_list) != 0:
            for radar_object in self.radar_object_list:
                
                if self.is_in_bbox(bbox, self.transform(radar_object)) == True:
                    self.filtered_radar_object_list.append(radar_object)
                    # cv2.line(self.image, self.transform(radar_object), self.transform(radar_object), (0, 255, 0), 15)
        
        min_iou = math.inf
        min_idx = 0
        cnt = 0

        if len(self.filtered_radar_object_list) != 0:
            for radar_object in self.filtered_radar_object_list:
                if math.sqrt((radar_object.x - camera_object[0])**2 + (radar_object.y - camera_object[1])**2) < min_iou:
                    min_idx = cnt
                    min_iou = math.sqrt((radar_object.x - camera_object[0])**2 + (radar_object.y - camera_object[1])**2)
                
                cnt += 1
        
            cv2.line(self.image, self.transform(self.filtered_radar_object_list[min_idx]), self.transform(self.filtered_radar_object_list[min_idx]), (0, 255, 0), 15)
        
            if camera_object[0] > self.filtered_radar_object_list[min_idx].x:
                final_distance = self.filtered_radar_object_list[min_idx].x
             
            else:
                final_distance = (camera_object[0] + self.filtered_radar_object_list[min_idx].x) / 2
            
            final_velocity = self.filtered_radar_object_list[min_idx].velocity

        else:
            final_distance = camera_object[0]
            # 카메라 속도 구현 설정해야함
            final_velocity = 5 / 3.6
        # print("Real distance : 15.52m")
        print("Camera distance : ", camera_object[0])
        only_camera_distance_list.append(camera_object[0])
        # print("Radar distance : ", self.filtered_radar_object_list[min_idx].x)
        only_radar_distance_list.append(self.filtered_radar_object_list[min_idx].x)
        # print("Distance : ", final_distance)
        fusion_distance_list.append(final_distance)
        # print("Velocity[km/h] : ", final_velocity * 3.6)
        velocity_list.append(final_velocity)
        
        print("-----------------------")

        # self.risk_calculate(final_distance, final_velocity)


    def risk_calculate(self, distance, velocity):
        
        if distance < 5:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
        
        else:
            car_velocity = self.my_speed + velocity

            crash_time = distance * 3600 / (1000 * (car_velocity - math.sin(math.radians(85))*self.my_speed))

            print("Crash time : ", crash_time)
            
            lane_change_time = 3.5 * 3600 / (1000*self.my_speed * math.cos(math.radians(85)))

            print("Lane change time : ", lane_change_time)

            print("-----------------------------------------------")
            
            # Ok to change lane
            if crash_time - lane_change_time >= 2.5 or self.my_speed > car_velocity:
                pass
            
            # Warning
            elif crash_time - lane_change_time < 2.5 and crash_time - lane_change_time >= 1.5:
                cv2.rectangle(self.image, (0, 0), (1280, 960), (0,130,255), 50, 1)
            
            # Dangerous
            else:
                cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)

    def visualize(self, data):

        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # print(self.image.shape)
        
        # self.inverse_transform()
        # self.transform()
        self.transformation_demo()

        cv2.imshow("Display", self.image)
        cv2.waitKey(1)

if __name__ == '__main__':
    only_camera_distance_list = []
    only_radar_distance_list = []
    fusion_distance_list = []
    velocity_list = []
    if not rospy.is_shutdown():
        fusion()
        rospy.spin()
    
    
    os.chdir('/home/heven/CoDeep_ws/src/yolov5_ros/src/csv')
    # df = pd.DataFrame({'Camera': only_camera_distance_list, 'Radar': only_radar_distance_list, 'Fusion': fusion_distance_list, 'Distance' : velocity_list})        

    # df.to_csv("distance.csv", index=True)

    # df = pd.DataFrame({'Fusion' : fusion_distance_list})
    df2 = pd.DataFrame({'Velocity' : velocity_list})

    # df.to_csv("fusion_distance.csv", index=False)
    df2.to_csv("veclocity.csv", index=False)