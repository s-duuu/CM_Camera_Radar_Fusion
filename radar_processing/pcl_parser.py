import rospy
import pcl
import pcl_helper
import math

from sensor_msgs.msg import PointCloud2
from dataclasses import dataclass

@dataclass
class object:
    index: int = None
    distance: int = None
    azimuth: int = None

class pcl_data_calc():
    def __init__(self):
        rospy.init_node('PclParser', anonymous=False)
        rospy.Subscriber('pointcloud/radar', PointCloud2, self.pcl_callback)
    
    def pcl_callback(self, data):
        raw_data = data
        self.radar_object_list = []
        cnt = 0
        """
        
        !!!Input PCL processing Code!!!
        
        """
        