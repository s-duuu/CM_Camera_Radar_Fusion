#!/usr/bin/env python

import rospy
import pcl
import math
import pcl_helper

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from yolov5_ros.msg import RadarObject
from yolov5_ros.msg import RadarObjectList

class pcl_data_calc():
    def __init__(self):
        rospy.init_node('PclParser', anonymous=False)
        rospy.Subscriber('/pointcloud/radar', PointCloud2, self.pcl_callback)

        self.radar_filtered_pub = rospy.Publisher('/pointcloud/filtered', PointCloud2, queue_size=1)
        self.radar_object_pub = rospy.Publisher('radar_objects', RadarObjectList, queue_size=1)
        
    
    def pcl_callback(self, data):
        # print("---------Data----------")
        # print(data)
        # Parameters for ROI setting and Removing noise
        self.xmin = 3.0
        self.xmax = 4.0
        self.mean_k = 1
        # 파라미터 수정
        self.thresh = 0.0003
        self.raw_list = []
        self.velocity_list = []
        self.cnt = 0
        # Convert sensor_msgs/PointCloud2 -> pcl
        cloud = pcl_helper.ros_to_pcl(data)
        
        for cloud_data in cloud:
            self.raw_list.append([cloud_data[0], cloud_data[1], cloud_data[2]])
            self.velocity_list.append(cloud_data[3])
            # print(cloud_data[3]*1000/3600)

        
        # print("-----------")
        
        if cloud.size > 0:

            # ROI setting
            cloud = self.do_passthrough(cloud, 'x', 17, 19)
            cloud = self.do_passthrough(cloud, 'y', 2.25, 3)
            # 변경 사항 시작
            # Objects = RadarObjectList()
            # for point_data in cloud:
            #     x = point_data[0]
            #     y = point_data[1]
            #     z = point_data[2]
            #     velocity = point_data[3]
                
            #     object_variable = RadarObject()
            #     object_variable.x = x
            #     object_variable.y = y
            #     object_variable.z = z
            #     object_variable.velocity = velocity

            #     Objects.RadarObjectList.append(object_variable)
            
            # self.radar_object_pub.publish(Objects)
            # 변경 사항 끝

            if cloud.size > 0:
                # Removing noise
                xyz_cloud = pcl_helper.XYZRGB_to_XYZ(cloud)

                # xyz_cloud = self.do_moving_least_squares(xyz_cloud)
                
                if xyz_cloud.size > 0:
                    xyz_cloud = self.do_statistical_outlier_filtering(xyz_cloud, self.mean_k, self.thresh)
                    
                    if xyz_cloud.size > 0:
                        xyz_cloud, _ = self.do_euclidean_clustering(xyz_cloud)
                    
                # Removing ground
                # _, _, cloud = self.do_ransac_plane_normal_segmentation(cloud, 0.05)
                Objects = RadarObjectList()

                cloud = pcl_helper.XYZ_to_XYZRGB(xyz_cloud, self.raw_list, self.velocity_list)

                # Converting into radar object message type
                for filtered_data in cloud:
                    x = filtered_data[0]
                    y = filtered_data[1]
                    z = filtered_data[2]
                    velocity = filtered_data[3]

                    print("x : ", x)
                    print("y : ", y)
                    print("z : ", z)
                    print("Velocity [km/h] : ", velocity*3.6)

                    # print("x type : ", type(x))
                    # print("y type : ", type(y))
                    # print("z type : ", type(z))
                    # print("velocity type : ", type(velocity))
                    # print(velocity*1000/3600)

                    # file.write("%f %f %f %f\n" % (x, y, z, velocity))

                    # index = self.cnt
                    # distance = x
                    # azimuth = math.atan(y/x)*180/math.pi

                    object_variable = RadarObject()
                    object_variable.x = x
                    object_variable.y = y
                    object_variable.z = z
                    object_variable.velocity = velocity

                    Objects.RadarObjectList.append(object_variable)

                    self.cnt += 1

                self.radar_object_pub.publish(Objects)
                # print(Objects)
            print("===============")
            # Convert pcl -> sensor_msgs/PointCloud2
            new_data = pcl_helper.pcl_to_ros(cloud)
            self.radar_filtered_pub.publish(new_data)
            rospy.loginfo("Filtered Point Published")
            # print("---Check---")
            # print(new_data)
        
            

        else:
            pass
    
    def do_passthrough(self, pcl_data, filter_axis, axis_min, axis_max):
        
        passthrough = pcl_data.make_passthrough_filter()
        passthrough.set_filter_field_name(filter_axis)
        passthrough.set_filter_limits(axis_min, axis_max)

        return passthrough.filter()
    
    
    def do_statistical_outlier_filtering(self, pcl_data, mean_k, thresh):
        
        outlier_filter = pcl_data.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(mean_k)
        outlier_filter.set_std_dev_mul_thresh(thresh)

        return outlier_filter.filter()
    
    def do_euclidean_clustering(self, pcl_data):

        tree = pcl_data.make_kdtree()

        ec = pcl_data.make_EuclideanClusterExtraction()

        # 점 사이 거리 (cm)
        ec.set_ClusterTolerance(0.01)
        # 점 개수
        ec.set_MinClusterSize(1)
        # 점 개수
        ec.set_MaxClusterSize(4)
        
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        color_cluster_point_list = []

        for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                color_cluster_point_list.append([pcl_data[indice][0],
                                                pcl_data[indice][1],
                                                pcl_data[indice][2]
                                                ])

        cluster_cloud = pcl.PointCloud()
        cluster_cloud.from_list(color_cluster_point_list)

        return cluster_cloud,cluster_indices



    def do_moving_least_squares(self, pcl_data):
        
        tree = pcl_data.make_kdtree()

        mls = pcl_data.make_moving_least_squares()
        mls.set_Compute_Normals(True)
        mls.set_polynomial_fit(True)
        mls.set_Search_Method(tree)
        # 파라미터 수정 (m 단위)
        mls.set_search_radius(10)
        # print('set parameters')
        mls_points = mls.process()

        return mls_points


    
    # def do_ransac_plane_normal_segmentation(self, pcl_data, input_max_distance):

    #     segmenter = pcl_data.make_segmenter_normals(ksearch=50)
    #     segmenter.set_optimize_coefficients(True)
    #     segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  #pcl_sac_model_plane
    #     segmenter.set_normal_distance_weight(0.1)
    #     segmenter.set_method_type(pcl.SAC_RANSAC) #pcl_sac_ransac
    #     segmenter.set_max_iterations(1000)
    #     segmenter.set_distance_threshold(input_max_distance) #0.03)  #max_distance
    #     indices, coefficients = segmenter.segment()

    #     inliers = pcl_data.extract(indices, negative=False)
    #     outliers = pcl_data.extract(indices, negative=True)

    #     return indices, inliers, outliers

    
    

if __name__ == '__main__':
    try:
        # file = open("/home/heven/CoDeep_ws/src/yolov5_ros/scripts/radar_processing/data", 'w')
        pcl_data_calc()
        rospy.spin()
        # file.close()
    
    except rospy.ROSInterruptException:
        pass
