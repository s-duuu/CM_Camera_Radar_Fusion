import rospy
import math

from msg import BoundingBoxes
from detector import Yolov5Detector
from dataclasses import dataclass

@dataclass
class object:
    index: int = None
    distance: float = None
    azimuth: float = None
    confidence: float = None

@dataclass
class boundingbox:
    xmin: int = None
    ymin: int = None
    xmax: int = None
    ymax: int = None
    

class image_data_calc(Yolov5Detector):
    def __init__(self):
        rospy.init_node('ImageParser', anonymous=False)
        rospy.Subscriber('yolov5/detections', BoundingBoxes, self.image_callback)
        
        
    def image_callback(self, data):
        
        self.camera_height = 1.0
        self.lane_width = 3.5
        self.horizontal_FOV = 50
        self.vertical_FOV = 28
        self.car_width = 1.97
        self.rear_to_camera = 2.5
        self.camera_object_list = []
        self.bounding_box_list = []
        
        bbox_list = data.bounding_boxes
        cnt = 0
        self.conf_threshold = 0.8
        
        # Image Distance & Azimuth Calculation
        for bbox in bbox_list:
            if bbox.probability > self.conf_threshold:
                camera_object = object()
                camera_object.index = cnt
                direct_distance = self.image_distance_calc(bbox)
                azimuth = self.image_azimuth_calc(bbox, direct_distance)
                distance = direct_distance * math.cos(azimuth * math.pi / 180) - self.rear_to_camera
                camera_object.distance = distance
                camera_object.azimuth = azimuth
                camera_object.confidence = bbox.probability
                
                bounding_box = boundingbox()
                bounding_box.xmin = bbox.xmin
                bounding_box.ymin = bbox.ymin
                bounding_box.xmax = bbox.xmax
                bounding_box.ymax = bbox.ymax
                
                self.camera_object_list.append(camera_object)
                self.bounding_box_list.append(bounding_box)
                
                cnt += 1

        
    
    def image_distance_calc(self, bbox):
        xmin = bbox.xmin
        xmax = bbox.xmax
        ymin = bbox.ymin
        ymax = bbox.ymax
        
        width = abs(xmax - xmin)
        height = abs(ymax - ymin)
        
        # Position-based distance estimation
        distance = self.camera_height * math.tan((math.pi/2 - math.atan((self.image.shape[0]/2 - (self.image.shape[0] - ymax)) * 2*math.tan(self.vertical_FOV*math.pi/360)/self.image.shape[0])))
        
        return distance
    
    def image_azimuth_calc(self, bbox, distance):
        xmin = bbox.xmin
        xmax = bbox.xmax
        ymin = bbox.ymin
        ymax = bbox.ymax
        
        azimuth = math.asin((self.lane_width / distance)) * 180 / math.pi
        
        return azimuth
        
        """
        
        !!!Input Azimuth Calculation Code!!!
        
        """
        

if __name__ == '__main__':
    try:
        image_data_calc()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass