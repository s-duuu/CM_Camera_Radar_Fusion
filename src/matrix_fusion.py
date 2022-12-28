#!/usr/bin/env python

import rospy
import os
import numpy as np
import numpy.linalg as lin
import kalman_filter
import math
import cv2
import pandas as pd

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from yolov5_ros.msg import BoundingBoxes
from yolov5_ros.msg import BoundingBox
from yolov5_ros.msg import CameraObjectList
from yolov5_ros.msg import RadarObjectList
from filterpy.kalman import KalmanFilter

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
        # rospy.Subscriber("/carmaker_vds_client_node/image_raw/compressed", CompressedImage, self.visualize)

    def camera_object_callback(self, data):
        self.bounding_box_list = data.bounding_boxes
        # rospy.Subscriber('radar_objects', RadarObjectList, self.radar_object_callback)

    def radar_object_callback(self, data):
        now = rospy.get_rostime()
        rospy.loginfo("Time : %i", now.secs)
        self.radar_object_list = data.RadarObjectList
        
    
    def is_in_bbox(self, bbox, radar_2d):
        
        if radar_2d[0] > bbox.xmin and radar_2d[0] < bbox.xmax and radar_2d[1] > bbox.ymin and radar_2d[1] < bbox.ymax:
            return True
        
        else:
            return False
    
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
        
        # YOLO detecting 될 때
        if len(self.bounding_box_list) != 0:
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

        # YOLO detecting 끊겼을 때
        else:

            min_x = math.inf
            for radar_object in self.radar_object_list:
                if radar_object.x < min_x:
                    min_x = radar_object.x
            
            self.radar_risk_calculate(min_x)
            

    # def first_matching(self, camera_object):

    #     if len(self.radar_object_list) != 0:
    #         for radar_object in self.radar_object_list:
    #             cv2.line(self.image, self.transform(radar_object), self.transform(radar_object), (0, 255, 0), 15)
    #             # print("point distance : ", (math.sqrt(((camera_object[0] - radar_object.x)**2) + ((camera_object[1] - radar_object.y)**2))))
    #             # print("point angle : ", math.degrees(abs(math.atan(camera_object[0]/camera_object[1]) - math.atan(radar_object.x/radar_object.y))))
    #             if (math.sqrt(((camera_object[0] - radar_object.x)**2) + ((camera_object[1] - radar_object.y)**2)) < self.distance_thresh) and math.degrees(abs(math.atan(camera_object[0]/camera_object[1]) - math.atan(radar_object.x/radar_object.y))) < self.angle_thresh:
    #                 self.filtered_radar_object_list.append(radar_object)

    #     min_iou = math.inf
    #     min_idx = 0
    #     cnt = 0

    #     if len(self.filtered_radar_object_list) != 0:
    #         for radar_object in self.filtered_radar_object_list:
                
    #             if math.sqrt((self.transform(radar_object)[0] - camera_object[0])**2 + (self.transform(radar_object)[1] - camera_object[1])**2) < min_iou:
    #                 min_idx = cnt
    #                 min_iou = math.sqrt((self.transform(radar_object)[0] - camera_object[0])**2 + (self.transform(radar_object)[1] - camera_object[1])**2)
                
    #             cnt += 1
        
    #         min_x = self.transform(self.filtered_radar_object_list[min_idx])[0]
    #         min_y = self.transform(self.filtered_radar_object_list[min_idx])[1]
        
    #         cv2.line(self.image, (min_x, min_y), (min_x, min_y), (0, 255, 0), 15)
        
    #         final_distance = camera_object[0]
    #         final_velocity = self.filtered_radar_object_list[min_idx].velocity
            
    #         print("Real distance : 15.52m")
    #         print("Distance : ", final_distance)
    #         # print("Radar distance : ", self.filtered_radar_object_list[min_idx].x)
    #         print("Velocity : ", final_velocity * 3.6)

    #         self.risk_calculate(final_distance, final_velocity)
    
    def second_matching(self, bbox, camera_object):
        
        # 레이더 데이터 있을 때
        if len(self.radar_object_list) != 0:
            for radar_object in self.radar_object_list:
                
                if self.is_in_bbox(bbox, self.transform(radar_object)) == True:
                    self.filtered_radar_object_list.append(radar_object)
        
        min_iou = math.inf
        min_idx = 0
        cnt = 0

        # 1단계 거친 레이더 포인트 남아있을 때
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

        # 레이더 포인트 없을 때
        else:
            final_distance = camera_object[0]
            final_velocity = -5.0 / 3.6


        # 속도 2번째 이후 loop
        if len(velocity_list) != 0:
            print("Initial : ", velocity_list[-1])
            kalman_velocity = kalman_filter.call_1dkalman(kf, velocity_list[-1], final_velocity)
        
        # 속도 1번째 loop
        else:
            kalman_velocity = final_velocity

        sum = 0
        total_num = len(self.radar_object_list)
        for radar_object in self.radar_object_list:
            sum += radar_object.x
        
        average = float(sum / total_num)
        only_radar_distance_list.append(average)
        
        only_camera_distance_list.append(camera_object[0])
        # print("Radar distance : ", self.filtered_radar_object_list[min_idx].x)
        # only_radar_distance_list.append(self.filtered_radar_object_list[min_idx].x)
        # print("Distance : ", final_distance)
        fusion_distance_list.append(final_distance)
        # print("Kalman Velocity[m/s] : ", kalman_velocity)
        velocity_list.append(kalman_velocity)
        
        print("-----------------------")

        self.risk_calculate(final_distance, kalman_velocity * 3.6)


    def risk_calculate(self, distance, velocity):
        
        if distance < 7:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
        
        else:
            car_velocity = self.my_speed - velocity

            crash_time = distance * 3600 / (1000 * (car_velocity - math.sin(math.radians(85))*self.my_speed))

            crash_list.append(crash_time)

            print("Crash time : ", crash_time)
            
            lane_change_time = 3.5 * 3600 / (1000*self.my_speed * math.cos(math.radians(85)))

            print("Lane change time : ", lane_change_time)

            print("-----------------------------------------------")
            
            # Ok to change lane
            if crash_time - lane_change_time >= 3.5 or self.my_speed > car_velocity:
                pass
            
            # Warning
            elif crash_time - lane_change_time < 3.5 and crash_time - lane_change_time >= 2.5:
                cv2.rectangle(self.image, (0, 0), (1280, 960), (0,130,255), 50, 1)
                cv2.line(self.image, (390, 745), (int((self.bounding_box_list[-1].xmin + self.bounding_box_list[-1].xmax)/2), self.bounding_box_list[-1].ymax), (0, 130, 255), 5, 1)
                cv2.line(self.image, (240, 745), (540, 745), (0, 130, 255), 5, 1)
                cv2.line(self.image, (self.bounding_box_list[-1].xmin, self.bounding_box_list[-1].ymax), (self.bounding_box_list[-1].xmax, self.bounding_box_list[-1].ymax), (0, 130, 255), 5, 1)

            # Dangerous
            else:
                cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
                cv2.line(self.image, (390, 745), (int((self.bounding_box_list[-1].xmin + self.bounding_box_list[-1].xmax)/2), self.bounding_box_list[-1].ymax), (0, 0, 255), 5, 1)
                cv2.line(self.image, (240, 745), (540, 745), (0, 0, 255), 5, 1)
                cv2.line(self.image, (self.bounding_box_list[-1].xmin, self.bounding_box_list[-1].ymax), (self.bounding_box_list[-1].xmax, self.bounding_box_list[-1].ymax), (0, 0, 255), 5, 1)
    
    def radar_risk_calculate(self, distance):

        if distance < 7:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
        
        elif distance < 12:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,130,255), 50, 1)
        
        else:
            pass

    def visualize(self, data):

        # self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        self.transformation_demo()

        cv2.imshow("Display", self.image)
        cv2.waitKey(1)

if __name__ == '__main__':
    
    kf = KalmanFilter(dim_x=2, dim_z=1)

    only_camera_distance_list = []
    only_radar_distance_list = []
    fusion_distance_list = []
    velocity_list = []
    crash_list = []

    if not rospy.is_shutdown():
        fusion()
        rospy.spin()
    
    
    os.chdir('/home/heven/CoDeep_ws/src/yolov5_ros/src/csv/result')

    df = pd.DataFrame({'Camera': only_camera_distance_list, 'Radar': only_radar_distance_list, 'Fusion': fusion_distance_list})        
    df.to_csv("distance_fusion_result.csv", index=True)

    # df2 = pd.DataFrame({'Velocity' : velocity_list})
    # df2.to_csv("velocity_fusion_result.csv", index=False)

    # df3 = pd.DataFrame({'Crash time' : crash_list})
    # df3.to_csv("crash_fusion_result.csv", index=False)

    # df4 = pd.DataFrame({'Radar': only_radar_distance_list})
    # df4.to_csv("only_radar_distance.csv", index=False)
