#!/usr/bin/env python

import rospy
import pcl
import math
import pcl_helper

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

class pcl_data_calc():
    def __init__(self):
        rospy.init_node('PclParser', anonymous=False)
        rospy.Subscriber('/pointcloud/radar', PointCloud2, self.pcl_callback)

        self.pub = rospy.Publisher('/pointcloud/filtered', PointCloud2, queue_size=1)
    
    def pcl_callback(self, data):
        
        # Parameters for ROI setting and Removing noise
        self.xmin = 3.0
        self.xmax = 4.0
        self.mean_k = 1
        self.thresh = 0.001
        for i in pc2.read_points(data, skip_nans=True):
            print(i)
        # Convert sensor_msgs/PointCloud2 -> pcl
        cloud = pcl_helper.ros_to_pcl(data)
        if cloud.size > 0:

            # ROI setting
            cloud = self.do_passthrough(cloud, 'y', -5, -1.75)

            if cloud.size > 0:
                # Removing noise
                cloud = pcl_helper.XYZRGB_to_XYZ(cloud)
                cloud = self.do_moving_least_squares(cloud)
                
                if cloud.size > 0:
                    cloud = self.do_statistical_outlier_filtering(cloud, self.mean_k, self.thresh)    
                # Removing ground
                # _, _, cloud = self.do_ransac_plane_normal_segmentation(cloud, 0.05)
                cloud = pcl_helper.XYZ_to_XYZRGB(cloud, (255,255,255))
            
            # Convert pcl -> sensor_msgs/PointCloud2
            
            new_data = pcl_helper.pcl_to_ros(cloud)
            self.pub.publish(new_data)
        
        else:
            pass
    
    def do_passthrough(self, pcl_data, filter_axis, axis_min, axis_max):
        
        passthrough = pcl_data.make_passthrough_filter()
        passthrough.set_filter_field_name(filter_axis)
        passthrough.set_filter_limits(axis_min, axis_max)

        return passthrough.filter()
    
    
    def do_statistical_outlier_filtering(self, pcl_data,mean_k,thresh):
        
        outlier_filter = pcl_data.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(mean_k)
        outlier_filter.set_std_dev_mul_thresh(thresh)

        return outlier_filter.filter()
    
    def do_euclidean_clustering(self, pcl_data):

        tree = pcl_data.make_kdtree()

        ec = pcl_data.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.015)
        ec.set_MinClusterSize(50)
        ec.set_MaxClusterSize(20000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()
        cluster_color = pcl_helper.get_color_list(len(cluster_indices))

        color_cluster_point_list = []

        for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                color_cluster_point_list.append([pcl_data[indice][0],
                                                pcl_data[indice][1],
                                                pcl_data[indice][2],
                                                pcl_helper.rgb_to_float(cluster_color[j])])

        cluster_cloud = pcl.PointCloud()
        cluster_cloud.from_list(color_cluster_point_list)

        return cluster_cloud,cluster_indices



    def do_moving_least_squares(self, pcl_data):
        
        tree = pcl_data.make_kdtree()

        mls = pcl_data.make_moving_least_squares()
        mls.set_Compute_Normals(True)
        mls.set_polynomial_fit(True)
        mls.set_Search_Method(tree)
        mls.set_search_radius(100)
        print('set parameters')
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
        pcl_data_calc()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass