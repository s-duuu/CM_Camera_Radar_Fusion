import rospy
import math

from msg import BoundingBoxes
from dataclasses import dataclass

@dataclass
class object:
    index: int = None
    distance: int = None
    azimuth: int = None
    confidence: float = None

@dataclass
class boundingbox:
    xmin: int = None
    ymin: int = None
    xmax: int = None
    ymax: int = None
    

class image_data_calc():
    def __init__(self):
        rospy.init_node('ImageParser', anonymous=False)
        rospy.Subscriber('yolov5/detections', BoundingBoxes, self.image_callback)
        
        
    def image_callback(self, data):
        bbox_list = data.bounding_boxes
        self.camera_object_list = []
        self.bounding_box_list = []
        cnt = 0
        self.conf_threshold = 0.8
        
        # Image Distance & Azimuth Calculation
        for bbox in bbox_list:
            if bbox.probability > self.conf_threshold:
                camera_object = object()
                camera_object.index = cnt
                camera_object.distance = self.iamge_distance_calc(bbox)
                camera_object.azimuth = self.image_azimuth_calc(bbox)
                camera_object.confidence = bbox.probability
                
                bounding_box = boundingbox()
                bounding_box.xmin = bbox.xmin
                bounding_box.ymin = bbox.ymin
                bounding_box.xmax = bbox.xmax
                bounding_box.ymax = bbox.ymax
                
                self.camera_object_list.append(camera_object)
                self.bounding_box_list.append(bounding_box)
                
                cnt += 1

        
    
    def iamge_distance_calc(self, bbox):
        xmin = bbox.xmin
        xmax = bbox.xmax
        ymin = bbox.ymin
        ymax = bbox.ymax
        
        width = abs(xmax - xmin)
        height = abs(ymax - ymin)
        
        """
        
        !!!Input Distance Calculation Code!!!
        
        """
    
    def image_azimuth_calc(self, bbox):
        xmin = bbox.xmin
        xmax = bbox.xmax
        ymin = bbox.ymin
        ymax = bbox.ymax
        
        width = abs(xmax - xmin)
        height = abs(ymax - ymin)
        
        """
        
        !!!Input Azimuth Calculation Code!!!
        
        """